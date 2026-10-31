import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Pre-compute some constants
    MAX_M_LIMIT = 200  # We won't go much higher than this in the first phase due to time
    
    def compute_score_fft(U):
        """
        Computes the score using FFT for speed.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
        """
        U = sorted(set(U))
        if not U or 0 not in U or len(U) < 2:
            return 0.0
        
        max_u = max(U)
        if max_u == 0:
            return 0.0
            
        # FFT size needs to be at least 2*max_u + 1 for linear convolution
        # To be safe and efficient, use next power of 2
        size = 1
        while size < 2 * max_u + 1:
            size *= 2
        
        # Create boolean array for U
        arr = np.zeros(size, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
            
        # FFT of U
        fft_u = np.fft.fft(arr)
        
        # Sumset: U+U corresponds to convolution of U with U
        # |U+U| is the number of non-zero entries in the convolution
        sumset_fft = fft_u * fft_u
        sumset_conv = np.fft.ifft(sumset_fft).real
        # Count non-zero entries (with small tolerance for numerical error)
        n_sum = np.sum(sumset_conv > 1e-6)
        
        # Difference set: U-U corresponds to convolution of U with reversed U
        # rev_arr[x] = 1 if -x is in U? No, simpler:
        # Convolution of A and B is sum A[i]B[j] for i+j=k.
        # For difference, we want a-b = k.
        # This is equivalent to convolution of A and B_rev where B_rev[i] = B[-i].
        # Since U contains non-negative integers, we can just compute the autocorrelation.
        # Autocorrelation R[k] = sum U[i] U[i+k]. This gives counts of pairs with difference k.
        # |U-U| is the number of k such that R[k] > 0.
        # Note: R[k] for k<0 is R[-k]. So we just need to check all k.
        
        # Actually, simpler: The support of the difference set is the support of the autocorrelation.
        # Autocorrelation via FFT:
        # corr = ifft(fft(arr) * conj(fft(arr)))
        # This gives circular correlation. Since we padded to size > 2*max, it's linear.
        corr_fft = fft_u * np.conj(fft_u)
        corr = np.fft.ifft(corr_fft).real
        
        # The correlation array corr[k] represents sum_{i} arr[i] arr[i-k] ?
        # Let's check indices.
        # Standard FFT autocorrelation:
        # c[k] = sum_i a[i] a[i-k]
        # The number of distinct differences is the number of non-zero c[k].
        # Due to symmetry, c[k] = c[-k].
        # We just need to count how many k have c[k] > 0.
        
        # Note: c[0] is always |U|.
        # We need to be careful with the index wrapping.
        # The result from ifft is in order 0, 1, ..., size-1.
        # Negative indices are at the end.
        
        n_diff = np.sum(corr > 1e-6)
        
        if n_sum == 0 or n_diff == 0:
            return 0.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def compute_score_naive(U):
        """Fallback for small sets or verification"""
        U = sorted(set(U))
        if not U or 0 not in U or len(U) < 2:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
            
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
                
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return 0.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    # Use FFT for larger sets, naive for very small to avoid overhead
    def get_score(U):
        if max(U) > 100:
            return compute_score_fft(U)
        else:
            return compute_score_naive(U)

    # Initial best
    best_set = [0, 1, 3]
    best_score = get_score(best_set)

    # Helper to generate a random set
    def random_set(n, max_m):
        if n <= 1:
            return [0]
        elems = [0] + random.sample(range(1, max_m + 1), min(n - 1, max_m))
        return sorted(set(elems))

    # GA Operators
    def crossover(p1, p2):
        # Simple uniform crossover
        p1 = list(p1)
        p2 = list(p2)
        # Pad to same length
        max_len = max(len(p1), len(p2))
        while len(p1) < max_len:
            p1.append(None)
        while len(p2) < max_len:
            p2.append(None)
            
        child = []
        for i in range(max_len):
            if random.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
                
        # Remove None and duplicates, ensure 0 is present
        child = [x for x in child if x is not None]
        if 0 not in child:
            child.append(0)
        return sorted(set(child))

    def mutate(U, max_m, prob=0.1):
        U = list(U)
        # Randomly change an element
        if random.random() < prob and len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            U[idx] = random.randint(1, max_m)
        # Randomly add an element
        if random.random() < prob * 0.5:
            U.append(random.randint(1, max_m))
        # Randomly remove an element (not 0)
        if random.random() < prob * 0.5 and len(U) > 2:
            idx = random.randint(1, len(U) - 1)
            U.pop(idx)
            
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
        return U

    # Main GA Loop
    population_size = 50
    max_generations = 200
    
    # Initialize population
    population = []
    scores = []
    
    # Seed with some structured sets
    # Arithmetic Progression
    for step in [1, 2, 3, 4, 5]:
        for length in [3, 4, 5, 6]:
            ap = list(range(0, step * length, step))
            if 0 in ap:
                population.append(ap)
                scores.append(get_score(ap))
                
    # Random sets
    while len(population) < population_size:
        n = random.randint(3, 10)
        m = random.randint(10, 50)
        s = random_set(n, m)
        population.append(s)
        scores.append(get_score(s))
        
    # Sort by score
    combined = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    scores, population = zip(*combined) if combined else ([], [])
    
    if scores:
        best_score = scores[0]
        best_set = population[0]
    else:
        best_score = 0
        best_set = [0, 1]

    # Evolution
    for gen in range(max_generations):
        if time.time() > deadline - 10:
            break
            
        # Elitism: keep top 10%
        elite_count = max(5, population_size // 10)
        elite = population[:elite_count]
        
        # Selection: Tournament
        def select():
            t1 = random.randint(0, len(population) - 1)
            t2 = random.randint(0, len(population) - 1)
            if scores[t1] > scores[t2]:
                return population[t1]
            else:
                return population[t2]
                
        new_population = list(elite)
        
        # Fill rest with offspring
        while len(new_population) < population_size:
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            child = mutate(child, 100, prob=0.2)
            
            # Constraint: Max element should not be too large to keep score competitive?
            # Actually, larger max U increases denominator.
            # But larger U can increase ratio.
            # Let's cap max element to 200 for now to keep FFT fast.
            if max(child) > 200:
                # Scale down or truncate?
                # Just truncate to keep it simple, or skip.
                # Let's just skip if it's too big, or scale.
                # Scaling is hard. Let's just re-sample if too big.
                if random.random() < 0.5:
                    child = random_set(len(child), 100)
                else:
                    # Truncate largest elements
                    child = sorted(child)[:10]
                    if max(child) > 200:
                         child = [0] + [x for x in child if x <= 200]
                         if len(child) < 2:
                             child = [0, 1]
            
            new_population.append(child)
            
        # Evaluate new population
        new_scores = [get_score(u) for u in new_population]
        
        # Combine and sort
        combined = sorted(zip(new_scores, new_population), key=lambda x: x[0], reverse=True)
        new_scores, new_population = zip(*combined) if combined else ([], [])
        
        # Update best
        if new_scores and new_scores[0] > best_score:
            best_score = new_scores[0]
            best_set = new_population[0]
            
        population = list(new_population)
        scores = list(new_scores)
        
        # Local search on the best individual every few generations
        if gen % 5 == 0 and time.time() < deadline - 20:
            # Try to improve best_set with a quick local search
            curr = list(best_set)
            curr_score = best_score
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                # Try adding
                if len(curr) < 15:
                    max_cur = max(curr)
                    for new_el in range(1, min(max_cur + 10, 201)):
                        if new_el not in curr:
                            cand = sorted(curr + [new_el])
                            s = get_score(cand)
                            if s > curr_score:
                                curr = cand
                                curr_score = s
                                improved = True
                                break
                # Try removing
                if len(curr) > 2:
                    for i in range(1, len(curr)):
                        cand = [x for j, x in enumerate(curr) if j != i]
                        s = get_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            improved = True
                            break
                # Try replacing
                if len(curr) > 1:
                    for i in range(1, len(curr)):
                        max_cur = max(curr)
                        for new_el in range(1, min(max_cur + 10, 201)):
                            if new_el not in curr:
                                cand = list(curr)
                                cand[i] = new_el
                                cand = sorted(set(cand))
                                if 0 in cand and len(cand) >= 2:
                                    s = get_score(cand)
                                    if s > curr_score:
                                        curr = cand
                                        curr_score = s
                                        improved = True
                                        break
                        if improved:
                            break
                            
            if curr_score > best_score:
                best_score = curr_score
                best_set = curr

    # Final verification
    # Ensure 0 is in best_set
    if 0 not in best_set:
        best_set.append(0)
    best_set = sorted(set(best_set))
    
    return best_set

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res)
