"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

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
    # This is equivalent to: for all k, bit k of wire[i] is bit i of k.
    
    def make_mask(i):
        # Create a big int with 1s at positions where bit i is set.
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        
        # Block: 2^i zeros, then 2^i ones.
        # Value of the 'ones' part: (1 << (1 << i)) - 1
        # Shifted by 2^i: block = ((1 << (1 << i)) - 1) << (1 << i)
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        
        # Geometric series factor to repeat the block
        # G = 1 + 2^L + 2^(2L) + ... + 2^((num_blocks-1)*L)
        # G = (2^(L*num_blocks) - 1) / (2^L - 1)
        # Note: 2^(L*num_blocks) = 2^TOTAL_INPUTS
        # Using integer division for exact power of 2 division
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

    # --- Phase 1: Greedy removal ---
    # With fast verification, we can do more passes.
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 10:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing idx
            candidate = network[:]
            candidate.pop(idx)
            if verify_network(candidate):
                network = candidate
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                removed = True
                break  # Restart since indices changed

    # --- Phase 2: Local-search with diverse moves ---
    # Incorporates move, swap, replace, and targeted replace moves from inspiration 1.
    # Also uses greedy removal after accepting a move to keep the network minimal.
    current = best_network[:]
    
    # Helper to perform greedy removal on a given network
    def greedy_remove(net):
        changed = True
        while changed and (time.time() - start_time) < DEADLINE - 2:
            changed = False
            for idx in range(len(net)):
                if (time.time() - start_time) > DEADLINE - 1:
                    break
                candidate = net[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    net = candidate
                    changed = True
                    break
        return net

    for iteration in range(5000): # Limit iterations to ensure time budget
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        r = random.random()
        n_comp = len(current)
        candidate = current[:]
        
        if r < 0.35 and n_comp > 1:
            # Move: Pick a comparator and move it to a new position
            i = random.randint(0, n_comp - 1)
            j = random.randint(0, n_comp - 1)
            comp = candidate.pop(i)
            if i < j:
                j -= 1
            j = max(0, min(j, len(candidate)))
            candidate.insert(j, comp)
        elif r < 0.65 and n_comp > 1:
            # Swap: Swap two comparators
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
            # Try wires near the old comparator
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
                best_network = candidate[:]
                best_size = len(candidate)
                current = candidate[:]
            else:
                # Accept if size is <= current size to allow exploration
                if len(candidate) <= len(current):
                    current = candidate[:]

    # --- Phase 3: Final Greedy Removal ---
    final_net = best_network[:]
    final_net = greedy_remove(final_net)
    
    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
