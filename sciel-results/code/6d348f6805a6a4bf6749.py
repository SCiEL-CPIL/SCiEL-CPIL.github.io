"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 55.0  # seconds, safety margin before 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification.
    Uses Python big integers to represent the state of each wire across all 2^18 inputs.
    min(a,b) is bitwise AND, max(a,b) is bitwise OR.
    Sorted check: (wire[i] & ~wire[i+1]) == 0 for all i.
    """
    start_time = time.time()

    # --- Initialize Bit-Parallel Wires ---
    # wire[i] is a Python int where bit k is 1 if input k has 1 on wire i.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    # So wire[i] has bits set at positions k where (k >> i) & 1 == 1.
    
    def make_mask(i):
        # Create a big int with 1s at positions where bit i is set.
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        
        # Geometric series factor G = 1 + 2^L + 2^(2L) + ...
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        # Copy the initial state
        wires = masks[:]
        
        for i, j in network:
            # min to i, max to j
            # min is AND, max is OR
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
            
        # Check sorted: wire[i] <= wire[i+1] for all i
        # This means no bit is 1 in wire[i] if it is 0 in wire[i+1].
        # So (wire[i] & ~wire[i+1]) must be 0.
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Verify initial network
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Helper: Aggressive Greedy Removal ---
    def greedy_remove(net):
        """
        Iteratively remove redundant comparators.
        Returns the reduced network.
        """
        current = net[:]
        removed = True
        # Limit time per greedy pass to avoid getting stuck
        greedy_start = time.time()
        
        while removed and (time.time() - greedy_start) < 2.0:
            removed = False
            # Scan from start to end
            for idx in range(len(current)):
                if (time.time() - start_time) > DEADLINE - 2:
                    return current
                candidate = current[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    current = candidate
                    removed = True
                    break # Restart scan as indices shifted
        
        return current

    # Apply initial aggressive reduction
    network = greedy_remove(network)
    if len(network) < best_size:
        best_network = network[:]
        best_size = len(network)

    # --- Phase 2: Local Search with Diverse Moves ---
    # Combines the structural exploration of Inspiration 1 (move/swap/replace)
    # with the fast verification and error-guided repair of the Current Program.
    
    current = best_network[:]
    
    # Set a random seed for reproducibility if needed, but randomness is good for escape
    random.seed(42)

    while (time.time() - start_time) < DEADLINE - 5:
        if len(current) <= 70:
            break
            
        n_comp = len(current)
        r = random.random()
        
        candidate = current[:]
        
        if r < 0.40:
            # Move: Pick a comparator and move it to a new position
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = candidate.pop(i)
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
        elif r < 0.70:
            # Swap: Swap two comparators
            if n_comp > 1:
                i, j = random.sample(range(n_comp), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.85:
            # Replace: Replace a comparator with a random one
            i = random.randint(0, n_comp - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            candidate[i] = (u, v)
        else:
            # Targeted replace: Replace with a comparator near an existing one
            i = random.randint(0, n_comp - 1)
            old_u, old_v = candidate[i]
            candidates = []
            for du in [-1, 0, 1]:
                for dv in [-1, 0, 1]:
                    nu = old_u + du
                    nv = old_v + dv
                    if 0 <= nu < N and 0 <= nv < N and nu != nv:
                        if nu > nv:
                            nu, nv = nv, nu
                        if (nu, nv) != (old_u, old_v):
                            candidates.append((nu, nv))
            if candidates:
                candidate[i] = random.choice(candidates)
            else:
                continue

        # Verify candidate
        if verify_network(candidate):
            # Accept and try to reduce
            candidate = greedy_remove(candidate)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_network = candidate[:]
                current = candidate
            else:
                # Accept if size is <= current size to allow exploration (plateau walking)
                if len(candidate) <= len(current):
                    current = candidate
        else:
            # If broken, try error-guided repair (from Current Program)
            # This helps escape local minima by fixing specific inversions.
            wires_err = masks[:]
            for i, j in candidate:
                new_wi = wires_err[i] & wires_err[j]
                new_wj = wires_err[i] | wires_err[j]
                wires_err[i] = new_wi
                wires_err[j] = new_wj
                
            inv_pair = None
            for i in range(N - 1):
                if (wires_err[i] & ~wires_err[i+1]) != 0:
                    inv_pair = (i, i+1)
                    break
                    
            if inv_pair:
                # Try inserting the fixing comparator at various positions
                # We limit attempts to save time
                positions = list(range(len(candidate) + 1))
                random.shuffle(positions)
                
                repaired = False
                for pos in positions[:10]: # Only try first 10 shuffled positions
                    if (time.time() - start_time) > DEADLINE - 5:
                        break
                    temp = candidate[:]
                    temp.insert(pos, inv_pair)
                    
                    if verify_network(temp):
                        # If repair works, we might have increased size, but it's a valid state.
                        # We can try to reduce it again.
                        temp = greedy_remove(temp)
                        if len(temp) < best_size:
                            best_size = len(temp)
                            best_network = temp[:]
                        current = temp
                        repaired = True
                        break

    # --- Phase 3: Final Greedy Removal ---
    final_net = best_network[:]
    final_net = greedy_remove(final_net)
    
    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
