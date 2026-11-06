import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    N = 512
    K = 32
    
    # Pre-compute "density" of each number in [1, 512]
    # Density is the number of 3-term APs it participates in as a, a+d, or a+2d
    # This helps initialize with numbers that are structurally "rich"
    density = [0] * (N + 1)
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            density[a] += 1
            density[b] += 1
            density[c] += 1
    
    # Sort indices by density (descending) to pick "best" seeds
    # We want to sample from the top 50% but with some randomness
    sorted_indices = sorted(range(1, N + 1), key=lambda x: density[x], reverse=True)
    top_half = sorted_indices[:N // 2]
    bottom_half = sorted_indices[N // 2:]
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set (bit x-1 for x)."""
        count = 0
        # d ranges 1..255 (since max element 512, 2d <= 511)
        # Actually max d is floor((512-1)/2) = 255
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
        
    def set_to_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
        
    def bitset_to_set(b):
        return {i + 1 for i in range(N) if (b >> i) & 1}

    def generate_initial_set():
        """Generate a set biased towards high-density numbers."""
        s = set()
        # Pick 80% from top half, 20% from anywhere to ensure diversity
        num_top = int(K * 0.8)
        num_rest = K - num_top
        
        # Sample without replacement from top_half
        s.update(random.sample(top_half, num_top))
        
        # Sample from remaining numbers
        remaining = [x for x in range(1, N + 1) if x not in s]
        s.update(random.sample(remaining, num_rest))
        
        return s

    # Population based search
    pop_size = 50
    population = []
    best_set_global = None
    best_score_global = -1
    
    # Initialize population
    for _ in range(pop_size):
        s = generate_initial_set()
        b = set_to_bitset(s)
        sc = score_bitset(b)
        population.append((sc, s, b))
        
        if sc > best_score_global:
            best_score_global = sc
            best_set_global = s

    # Sort population by score
    population.sort(key=lambda x: x[0], reverse=True)
    
    # Iterative improvement
    # Each iteration:
    # 1. Take the top half of the population as parents
    # 2. Generate children via mutation (swap elements)
    # 3. Keep the top half of the combined pool (parents + children)
    
    max_iterations = 1000
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Select parents: top 50%
        parents = population[:pop_size // 2]
        
        new_population = []
        
        # Create children from parents
        for parent_score, parent_set, parent_b in parents:
            # Create 2 children per parent
            for _ in range(2):
                child_set = set(parent_set)
                child_b = parent_b
                
                # Mutation: Swap 2 random elements with 2 new random elements
                # This allows larger jumps than single element replacement
                num_swaps = random.choice([1, 2])
                
                for _ in range(num_swaps):
                    if len(child_set) < 2:
                        break
                    # Pick two elements to remove
                    elems_to_remove = random.sample(list(child_set), 2)
                    for e in elems_to_remove:
                        child_set.remove(e)
                        child_b &= ~(1 << (e - 1))
                        
                    # Pick two new elements
                    available = [x for x in range(1, N + 1) if x not in child_set]
                    if len(available) < 2:
                        # Restore if we broke it (shouldn't happen with N=512, K=32)
                        break
                    new_elems = random.sample(available, 2)
                    for e in new_elems:
                        child_set.add(e)
                        child_b |= (1 << (e - 1))
                        
                child_score = score_bitset(child_b)
                new_population.append((child_score, child_set, child_b))
                
                # Update global best
                if child_score > best_score_global:
                    best_score_global = child_score
                    best_set_global = child_set
        
        # Combine parents and children, sort by score, keep top pop_size
        # We keep the best parents and the best children
        # Actually, standard evolutionary approach:
        # Pool = parents + children
        # Sort Pool
        # New Population = Top pop_size
        
        # To save time, we don't re-evaluate parents.
        # But we need to merge them.
        # Since parents are already scored, we can just concatenate and sort.
        # However, sorting 50 + 50 = 100 items is cheap.
        
        # Note: We are discarding the bottom half of parents.
        # This is "elitism" if we kept them, but here we are doing generational replacement
        # with selection pressure.
        
        # Let's add a bit of elitism: keep the single best individual always
        # The global best is tracked separately, so we don't strictly need to keep it in pop,
        # but it helps if the search stagnates.
        
        all_candidates = parents + new_population
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top pop_size
        population = all_candidates[:pop_size]
        
        # If we found a new global best, inject it into the population to ensure it's considered
        # (It's already there if it was in new_population or parents, but good for safety)
        # Actually, if best_set_global was found in new_population, it's in all_candidates.
        # If it was from a previous generation, it might have been dropped if not in top 50.
        # To prevent loss of the global best, let's force it into the population.
        if best_set_global:
            best_b = set_to_bitset(best_set_global)
            # Check if it's already in population
            found = False
            for i, (sc, s, b) in enumerate(population):
                if s == best_set_global:
                    found = True
                    break
            if not found:
                # Replace the worst individual in population
                population[-1] = (best_score_global, best_set_global, best_b)
                population.sort(key=lambda x: x[0], reverse=True)

    return sorted(best_set_global)

def get_score(S):
    S_set = set(S)
    S_sorted = sorted(S)
    d_set = set()
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i+1, n):
            b = S_sorted[j]
            d = b - a
            c = b + d
            if c in S_set:
                d_set.add(d)
    return len(d_set)
