"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64
    words_per_wire = NUM_WORDS

    # Pre-compute initial wires
    initial_wires = [None] * N
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators):
        """Try to remove comparators one by one, restarting from the end."""
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break
        return current

    def local_search(comparators):
        """Perform local search by replacing and removing comparators."""
        current = list(comparators)
        random.seed(42)
        
        for iteration in range(500):
            if time.time() > deadline - 2.0:
                break
            if not current:
                break
            
            idx = random.randint(0, len(current) - 1)
            old_i, old_j = current[idx]
            
            # Try a few random replacements
            found = False
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i == new_j:
                    continue
                if (new_i, new_j) == (old_i, old_j):
                    continue
                
                test = list(current)
                test[idx] = (new_i, new_j)
                
                if verify_network(test):
                    current = test
                    found = True
                    break
            
            if found:
                # Quick removal pass
                current = greedy_remove(current)
        
        return current

    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)

    # Phase 1: Greedy removal
    current = greedy_remove(best_network)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Phase 2: Local search
    current = local_search(best_network)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
