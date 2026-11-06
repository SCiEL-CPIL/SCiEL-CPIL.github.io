"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 56.0  # Safety margin before 60s kill

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096 words of 64 bits each

    # Pre-compute initial wires using efficient bit packing
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, max_iterations=50):
        """
        Iteratively remove redundant comparators with batch removal.
        Multiple passes to find all removable comparators.
        """
        current = list(comparators)
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline - 1.0:
                break
            n = len(current)
            if n == 0:
                break
            
            # Try batch removal: find all removable comparators in one pass
            removable = []
            for idx in range(n):
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    removable.append(idx)
            
            if removable:
                # Remove all identified redundant comparators at once
                removable_set = set(removable)
                current = [c for i, c in enumerate(current) if i not in removable_set]
                changed = True

        return current

    # Start with odd-even transposition
    base_network = odd_even_transposition()
    base_network = greedy_remove(base_network)

    best_network = list(base_network)
    best_count = len(base_network)

    random.seed(42)

    # Max-SAT inspired stochastic local search with simulated annealing
    # Variables: presence/absence of each comparator
    # Hard constraints: sorting correctness for all 2^18 inputs
    # Soft objective: minimize number of comparators (Max-SAT style)
    
    temperature = 5.0  # Initial temperature for simulated annealing
    cooling_rate = 0.995
    min_temperature = 0.5
    
    while time.time() < deadline - 2.0:
        if temperature < min_temperature:
            temperature = min_temperature
        
        current = list(base_network)
        n_comp = len(current)
        
        if n_comp == 0:
            break
        
        r = random.random()
        
        if r < 0.40:
            # Remove a comparator (Max-SAT: try to satisfy fewer soft clauses)
            if n_comp > 1:
                idx = random.randint(0, n_comp - 1)
                candidate = current[:idx] + current[idx + 1:]
                if verify_network(candidate):
                    # Accept the removal
                    current = candidate
                    current = greedy_remove(current)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_network = list(current)
                        base_network = list(current)
                    continue
        elif r < 0.65:
            # Move: Pick a comparator and move it to a new position
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = current[i]
                candidate = current[:i] + current[i+1:]
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
                if verify_network(candidate):
                    candidate = greedy_remove(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                    else:
                        # Simulated annealing: accept worse solutions with probability
                        if len(candidate) <= len(current) or random.random() < np.exp(-(len(candidate) - len(current)) / temperature):
                            base_network = list(candidate)
                    continue
        elif r < 0.85:
            # Replace: Replace a comparator with a different one
            idx = random.randint(0, n_comp - 1)
            old_u, old_v = current[idx]
            # Try to find a replacement that maintains sorting
            found = False
            for _ in range(20):  # Try multiple replacements
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) == (old_u, old_v):
                    continue
                candidate = list(current)
                candidate[idx] = (u, v)
                if verify_network(candidate):
                    candidate = greedy_remove(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
                    found = True
                    break
            if not found:
                # Try targeted replacement near existing comparator
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
                    for (u, v) in random.sample(candidates, min(5, len(candidates))):
                        candidate = list(current)
                        candidate[idx] = (u, v)
                        if verify_network(candidate):
                            candidate = greedy_remove(candidate)
                            if len(candidate) < best_count:
                                best_count = len(candidate)
                                best_network = list(candidate)
                                base_network = list(candidate)
                            break
        else:
            # Random restart: perturb the base network significantly
            if n_comp > 3:
                # Remove 1-3 random comparators and verify
                num_remove = random.randint(1, min(3, n_comp - 1))
                indices_to_remove = random.sample(range(n_comp), num_remove)
                indices_to_remove.sort(reverse=True)
                candidate = list(current)
                for idx in indices_to_remove:
                    candidate.pop(idx)
                if verify_network(candidate):
                    candidate = greedy_remove(candidate)
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
                        base_network = list(candidate)
        
        # Cool down
        temperature *= cooling_rate
    
    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
