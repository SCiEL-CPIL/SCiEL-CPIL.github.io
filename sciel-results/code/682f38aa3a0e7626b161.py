"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = np.array(sorted(set(U)), dtype=int)
        n = len(U)
        if n == 0 or U[0] != 0:
            return 0.0
        if n == 1:
            return 1.0
        
        # Compute sumset and difference set sizes
        # For large N, this is the bottleneck. 
        # We assume N is kept reasonable (< 200) for speed in SA loop.
        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        if n_sums == 0:
            return 0.0
            
        ratio = n_diffs / n_sums
        if ratio <= 0:
            return 0.0
            
        max_u = U[-1]
        if max_u == 0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
            
        return 1 + math.log(ratio) / denom

    def sidon_set(n):
        """Generate a Sidon-like set (distinct sums) of size n+1 starting with 0."""
        S = [0]
        for i in range(1, n + 1):
            # Greedy: find smallest x not in S such that no two sums collide
            # Simplified: use quadratic growth i^2 which is a Sidon set
            S.append(i * i)
        return S

    def random_set(n, max_val):
        """Generate a random set of size n containing 0."""
        if n == 1:
            return [0]
        others = random.sample(range(1, max_val + 1), n - 1)
        return sorted([0] + others)

    def simulated_annealing(initial_set, max_iters=5000, T0=100.0, alpha=0.9995, max_val=500):
        """Run Simulated Annealing to optimize the set."""
        current = list(initial_set)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        
        T = T0
        for i in range(max_iters):
            T *= alpha
            if T < 1e-6:
                break
            
            # Generate a neighbor
            r = random.random()
            new_set = list(current)
            
            if r < 0.4:
                # Add an element
                if len(new_set) < 150: # Cap size to keep scoring fast
                    x = random.randint(1, max_val)
                    if x not in new_set:
                        new_set.append(x)
                    else:
                        continue
            elif r < 0.7:
                # Remove an element (keep 0)
                if len(new_set) > 2:
                    idx = random.randint(1, len(new_set) - 1)
                    new_set.pop(idx)
            elif r < 0.9:
                # Replace an element
                if len(new_set) > 2:
                    idx = random.randint(1, len(new_set) - 1)
                    old_val = new_set[idx]
                    x = random.randint(1, max_val)
                    if x not in new_set:
                        new_set[idx] = x
                        new_set.sort()
                    else:
                        continue
            else:
                # Jump: randomize a few elements
                if len(new_set) > 2:
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(1, len(new_set) - 1)
                        x = random.randint(1, max_val)
                        if x not in new_set:
                            new_set[idx] = x
                    new_set.sort()
            
            new_score = score_of(new_set)
            
            # Acceptance probability
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score
        
        return best, best_score

    # 1. Generate diverse seeds
    seeds = []
    
    # Small structured sets
    seeds.append([0, 1, 2, 4, 8, 16, 32])
    seeds.append(sidon_set(10))
    seeds.append(sidon_set(20))
    seeds.append(sidon_set(50))
    
    # Random sets of various sizes
    random.seed(42)
    for size in [5, 10, 20, 30, 50]:
        for _ in range(3):
            seeds.append(random_set(size, max_val=1000))
            
    # Prime based
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    seeds.append([0] + primes[:10])
    seeds.append([0] + primes[:20])

    best_set = seeds[0]
    best_score = score_of(best_set)
    
    # 2. Run SA on top seeds
    # Sort seeds by initial score to prioritize good starts
    scored_seeds = [(score_of(s), s) for s in seeds]
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Run SA on the top 5 seeds
    for _, seed in scored_seeds[:5]:
        result_set, result_score = simulated_annealing(seed, max_iters=2000, T0=50.0, alpha=0.999, max_val=2000)
        if result_score > best_score:
            best_score = result_score
            best_set = result_set

    # 3. Final local polish on the best set found
    # Try a few greedy improvements on the best set
    current = list(best_set)
    current_score = best_score
    improved = True
    while improved:
        improved = False
        # Try adding
        for x in range(1, 2000, 10):
            if x not in current:
                new_set = current + [x]
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
        if improved: continue
        # Try removing
        for i in range(1, len(current)):
            new_set = current[:i] + current[i+1:]
            if len(new_set) < 2: continue
            s = score_of(new_set)
            if s > current_score + 1e-12:
                current = new_set
                current_score = s
                improved = True
                break
        if improved: continue
        # Try replacing
        for i in range(1, len(current)):
            old = current[i]
            for x in range(1, 2000, 10):
                if x == old or x in current: continue
                new_set = current[:i] + [x] + current[i+1:]
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            if improved: break

    if current_score > best_score:
        best_set = current
        best_score = current_score

    return sorted(best_set)
# EVOLVE-BLOCK-END
