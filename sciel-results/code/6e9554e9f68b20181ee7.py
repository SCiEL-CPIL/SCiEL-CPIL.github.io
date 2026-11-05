import time
import random
import numpy as np
from scipy.optimize import differential_evolution
from scipy.fft import rfft, irfft

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    num_targets = max_sum - min_sum + 1
    
    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * (max_sum + 1):
        n_fft *= 2

    def get_counts_fast(A_set):
        """
        Fast count of unordered pair sums for a set A.
        Returns a numpy array of counts for sums 0..max_sum.
        """
        f = np.zeros(max_sum + 1, dtype=float)
        for x in A_set:
            if 0 <= x <= max_sum:
                f[x] = 1.0
        
        F = rfft(f, n=n_fft)
        conv = irfft(F * F, n=n_fft)
        
        counts = np.zeros(max_sum + 1, dtype=int)
        for s in range(max_sum + 1):
            ordered = int(round(conv[s]))
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= N and half in A_set:
                    unordered = (ordered + 1) // 2
                else:
                    unordered = ordered // 2
            else:
                unordered = ordered // 2
            counts[s] = unordered
        return counts

    def get_cost(A_set):
        """
        Calculate the cost: 
        1. If any target sum is uncovered, return 1000 + number of uncovered.
        2. Otherwise, return the max multiplicity in the target range.
        """
        counts = get_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum+1]
        
        if np.any(target_counts == 0):
            uncovered_count = int(np.sum(target_counts == 0))
            return 1000 + uncovered_count
        
        return int(np.max(target_counts))

    def vector_to_set(vec):
        """Convert a binary vector to a set of indices."""
        return set(np.where(vec > 0.5)[0].tolist())

    def fitness(vec):
        """
        Fitness function for differential evolution.
        We want to minimize the peak multiplicity.
        To guide DE, we penalize uncovered sums heavily.
        """
        A_set = vector_to_set(vec)
        if len(A_set) == 0:
            return 10000
        return get_cost(A_set)

    def local_search(A_init, time_limit):
        """
        Fast local search to polish the solution.
        Tries adding, removing, and swapping elements to reduce peak or fix coverage.
        """
        A_set = set(A_init)
        current_cost = get_cost(A_set)
        best_cost = current_cost
        best_A = set(A_set)
        all_elements = set(range(N + 1))
        
        # Precompute counts for current set
        counts = get_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum+1]
        
        while time.time() < time_limit:
            # Check for improvement opportunities
            
            # 1. Try removing elements if it doesn't break coverage and reduces peak
            # 2. Try adding elements if it fixes coverage or reduces peak (unlikely for adding, usually increases)
            # 3. Try swapping
            
            improved = False
            
            # Heuristic: If we have coverage, try to reduce peak by swapping
            if current_cost < 1000:
                peak = current_cost
                if peak <= 2:
                    break
                
                # Find elements contributing to peak sums
                # Simplified: Just try random swaps and some targeted ones
                
                # Try swapping each element with a random candidate
                A_list = list(A_set)
                random.shuffle(A_list)
                
                for x_in in A_list:
                    if time.time() >= time_limit:
                        break
                    
                    # Try a few random swaps
                    candidates = list(all_elements - A_set)
                    if not candidates:
                        continue
                    
                    # Limit candidates to speed up
                    random.shuffle(candidates)
                    for x_out in candidates[:20]:
                        A_test = A_set.copy()
                        A_test.discard(x_in)
                        A_test.add(x_out)
                        
                        # Quick check: does this set cover all?
                        # We can approximate or do full check. Full check is ~1ms.
                        new_cost = get_cost(A_test)
                        
                        if new_cost < current_cost:
                            A_set = A_test
                            current_cost = new_cost
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)
                            break
                        elif new_cost == current_cost:
                            # Accept equal cost to escape local minima? 
                            # Only if we haven't improved in a while, but for simplicity, stick to strict improvement or slight randomization
                            pass

            # If coverage is broken, try to fix it
            elif current_cost >= 1000:
                uncovered = [s for s in target_sums if get_counts_fast(A_set)[s] == 0]
                # This is slow, better to use precomputed counts if we maintain them, 
                # but for simplicity in this local search, we recompute or accept the cost.
                # Actually, get_cost already returns 1000+.
                # Try adding an element that covers the most uncovered sums.
                
                # To save time, we'll just do random additions if coverage is bad
                candidates = list(all_elements - A_set)
                random.shuffle(candidates)
                for x_out in candidates[:10]:
                    A_test = A_set.copy()
                    A_test.add(x_out)
                    new_cost = get_cost(A_test)
                    if new_cost < current_cost:
                        A_set = A_test
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A_set)
                        break

            if not improved:
                # Try a random move to escape
                if random.random() < 0.5 and A_set:
                    x_in = random.choice(list(A_set))
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x_out = random.choice(candidates)
                        A_test = A_set.copy()
                        A_test.discard(x_in)
                        A_test.add(x_out)
                        new_cost = get_cost(A_test)
                        if new_cost <= current_cost:
                            A_set = A_test
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)
                else:
                    # Small random perturbation: flip a bit
                    if A_set and random.random() < 0.8:
                         # Try adding a random element
                         candidates = list(all_elements - A_set)
                         if candidates:
                             x_out = random.choice(candidates)
                             A_test = A_set.copy()
                             A_test.add(x_out)
                             new_cost = get_cost(A_test)
                             if new_cost <= current_cost:
                                 A_set = A_test
                                 current_cost = new_cost
                                 if current_cost < best_cost:
                                     best_cost = current_cost
                                     best_A = set(A_set)
                    elif not A_set:
                         # Add random
                         x_out = random.randint(0, N)
                         A_test = A_set.copy()
                         A_test.add(x_out)
                         A_set = A_test
                         current_cost = get_cost(A_set)
                         if current_cost < best_cost:
                             best_cost = current_cost
                             best_A = set(A_set)

        return best_A, best_cost

    # Strategy:
    # 1. Use Differential Evolution to find a good starting point.
    # 2. Polish with local search.
    
    random.seed(42)
    np.random.seed(42)
    
    best_overall_A = None
    best_overall_cost = float('inf')
    
    # DE Parameters
    # Bounds for each of the 129 variables (0 or 1)
    bounds = [(0, 1) for _ in range(N + 1)]
    
    # We run DE for a limited time or generations
    # Since DE can be slow, we run a few short iterations
    # or one iteration with limited generations.
    
    # Let's try 3 runs of DE with different populations
    de_configs = [
        {'popsize': 15, 'maxiter': 50, 'polish': False},
        {'popsize': 20, 'maxiter': 30, 'polish': False},
        {'popsize': 10, 'maxiter': 80, 'polish': False},
    ]
    
    for config in de_configs:
        if time.time() >= deadline - 15:
            break
            
        try:
            result = differential_evolution(
                fitness,
                bounds,
                maxiter=config['maxiter'],
                popsize=config['popsize'],
                seed=42,
                tol=1e-3,
                init='latinhypercube',
                polish=False,
                updating='immediate',
                callback=None
            )
            
            best_vec = result.x
            A_de = vector_to_set(best_vec)
            
            if len(A_de) == 0:
                continue
                
            # Polish the DE result
            A_polished, cost_polished = local_search(A_de, min(deadline - 10, time.time() + 10))
            
            if cost_polished < best_overall_cost:
                best_overall_cost = cost_polished
                best_overall_A = A_polished
                
            if best_overall_cost <= 4:
                break
                
        except Exception as e:
            pass

    # If DE didn't find a good solution, or as a backup, try a few random local searches
    if best_overall_cost > 6:
        for k in [25, 30, 35]:
            if time.time() >= deadline - 10:
                break
            A_init = set(random.sample(range(0, N + 1), k))
            A_refined, cost = local_search(A_init, min(deadline - 5, time.time() + 5))
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = A_refined

    # Final fallback
    if best_overall_A is None:
        # Construct a valid set, e.g., all even numbers or similar
        # A = {0, 2, ..., 128} covers even sums. 
        # We need all sums. 
        # A = [0, 128] covers 128. 
        # A = [0, 1, ..., 128] covers everything with high multiplicity.
        # Let's just return a random set that is likely valid if we are stuck, 
        # but the task requires validity. 
        # A simple valid set: A = [0, 128] U [64, 128] ?
        # Actually, A = [0, 128] is valid for 128. 
        # To cover [128, 256], we can use A = [0, 128]. Sums: 128 (0+128), 256 (128+128). 
        # This only covers 128 and 256.
        
        # A known small base:
        # A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128}
        # This is too big.
        
        # Let's use the result from the first local search if available, or a random one.
        A_init = set(random.sample(range(0, N + 1), 30))
        best_overall_A, best_overall_cost = local_search(A_init, deadline - 2)

    return list(best_overall_A)

if __name__ == "__main__":
    # Quick test
    A = construct()
    print(f"Found set of size {len(A)}")
    # Verify
    counts = np.zeros(257, dtype=int)
    for i, a in enumerate(A):
        for b in A[i:]:
            counts[a+b] += 1
    target_counts = counts[128:257]
    if np.any(target_counts == 0):
        print("Invalid: Uncovered sums found")
        print(np.where(target_counts == 0)[0] + 128)
    else:
        print(f"Valid. Peak multiplicity: {np.max(target_counts)}")
