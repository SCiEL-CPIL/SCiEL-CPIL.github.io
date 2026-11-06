"""
Size-efficient sorting network on 18 wires.
Combines bit-parallel verification (NumPy) with efficient local search and greedy reduction.
"""
import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 55.0  # seconds, safety margin before 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification with NumPy.
    Uses local search with diverse moves and efficient greedy removal.
    """
    start_time = time.time()

    # --- Initialize Bit-Parallel Wires using NumPy ---
    # We represent the state of all 2^18 inputs for each wire.
    # Since 2^18 = 262144 bits, and a uint64 has 64 bits, we need 262144 / 64 = 4096 words per wire.
    NUM_WORDS = TOTAL_INPUTS // 64
    
    # Pre-compute indices for all inputs
    indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    
    # Pre-compute weights for packing bits into uint64
    # We need to map bit positions 0..63 to weights 1, 2, 4, ... 2^63
    # Note: np.uint64 overflow is handled by wrapping, but we are doing sum of distinct powers of 2,
    # so no overflow occurs in the sum itself if we are careful, but standard integer arithmetic is fine.
    weights = 1 << np.arange(64, dtype=np.uint64)

    # Pre-compute initial wires
    # wire[i] is an array of NUM_WORDS uint64s.
    # Bit k of the logical input k corresponds to the k-th bit in our representation.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    # So for wire i, bit k is 1 if (k >> i) & 1 == 1.
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1  # Shape: (TOTAL_INPUTS,)
        bits_i_64 = bits_i.astype(np.uint64)
        # Reshape to (NUM_WORDS, 64) and pack
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(network):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        # Copy the initial state
        # Using a list of arrays. We must copy because we modify them.
        wires = [w.copy() for w in initial_wires]
        
        for i, j in network:
            wi = wires[i]
            wj = wires[j]
            # min is AND, max is OR
            wires[i] = wi & wj
            wires[j] = wi | wj
            
        # Check sorted: wire[i] <= wire[i+1] for all i
        # This means no bit is 1 in wire[i] if it is 0 in wire[i+1].
        # So (wire[i] & ~wire[i+1]) must be 0.
        for i in range(N - 1):
            if np.any(wires[i] & ~wires[i + 1]):
                return False
        return True

    def find_violation(network):
        """
        Returns the first adjacent inversion pair (i, i+1) if not sorted, else None.
        """
        wires = [w.copy() for w in initial_wires]
        for i, j in network:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
            
        for i in range(N - 1):
            if np.any(wires[i] & ~wires[i + 1]):
                return (i, i + 1)
        return None

    def greedy_remove(net, time_budget=1.0):
        """
        Iteratively remove redundant comparators.
        Uses systematic sampling to find removable comparators to avoid O(N^2) full scans.
        """
        current = net[:]
        start_greedy = time.time()
        removed = True
        
        while removed and (time.time() - start_greedy) < time_budget:
            removed = False
            n = len(current)
            if n == 0:
                break
            
            # Sample positions to check for removal
            # Bias towards ends and random middle
            positions = set()
            num_samples = min(20, n)
            for _ in range(num_samples):
                pos = random.randint(0, n - 1)
                positions.add(pos)
            
            # Always check start and end
            if n > 0:
                positions.add(0)
                positions.add(n - 1)
            
            for idx in positions:
                if idx >= len(current):
                    continue
                candidate = current[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    current = candidate
                    removed = True
                    break  # Restart sampling with new length
                    
        return current

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Verify initial network
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # Apply initial aggressive reduction
    network = greedy_remove(network, time_budget=5.0)
    if len(network) < best_size:
        best_network = network[:]
        best_size = len(network)

    # --- Phase 2: Local Search ---
    current = best_network[:]
    random.seed(42)

    while (time.time() - start_time) < DEADLINE - 5:
        if len(current) <= 70:
            break
            
        n_comp = len(current)
        r = random.random()
        
        candidate = current[:]
        valid_move = True
        
        if r < 0.35:
            # Move: Pick a comparator and move it to a new position
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = candidate.pop(i)
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
            else:
                valid_move = False
        elif r < 0.65:
            # Swap: Swap two comparators
            if n_comp > 1:
                i, j = random.sample(range(n_comp), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                valid_move = False
        elif r < 0.85:
            # Replace: Replace a comparator with a random one
            i = random.randint(0, n_comp - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                valid_move = False
            else:
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
                valid_move = False

        if not valid_move:
            continue

        # Verify candidate
        if verify_network(candidate):
            # Accept and try to reduce
            candidate = greedy_remove(candidate, time_budget=0.5)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_network = candidate[:]
                current = candidate
            else:
                # Accept if size is <= current size to allow exploration (plateau walking)
                if len(candidate) <= len(current):
                    current = candidate
        else:
            # If broken, try error-guided repair
            # Find the first violation
            inv_pair = find_violation(candidate)
            
            if inv_pair:
                # Try inserting the fixing comparator at various positions
                positions = list(range(len(candidate) + 1))
                random.shuffle(positions)
                
                repaired = False
                for pos in positions[:10]: # Only try first 10 shuffled positions
                    if (time.time() - start_time) > DEADLINE - 5:
                        break
                    temp = candidate[:]
                    temp.insert(pos, inv_pair)
                    
                    if verify_network(temp):
                        # If repair works, reduce it
                        temp = greedy_remove(temp, time_budget=0.5)
                        if len(temp) < best_size:
                            best_size = len(temp)
                            best_network = temp[:]
                        # Accept the repaired state as current to continue from here
                        current = temp
                        repaired = True
                        break

    # --- Phase 3: Final Greedy Removal ---
    final_net = best_network[:]
    final_net = greedy_remove(final_net, time_budget=5.0)
    
    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    # Final safety check
    if not verify_network(best_network):
        # Fallback to a known good network (odd-even transposition)
        # This should rarely happen if logic is correct, but prevents score 0
        fallback = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                fallback.append((i, i + 1))
        return fallback

    return best_network
# EVOLVE-BLOCK-END
