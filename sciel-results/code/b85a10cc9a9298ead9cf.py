import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Strategy:
    # 1. Generate a diverse set of high-quality initial candidates using algebraic structures.
    #    - Geometric progressions (dense in products)
    #    - Smooth numbers (dense in products, moderate in sums)
    #    - Arithmetic progressions (dense in sums, sparse in products - usually bad, but good for baseline)
    #    - Mixed structures
    # 2. Perform a fast local search using incremental delta evaluation.
    #    Instead of recomputing the full sum/prod sets (O(n^2)), we compute the change in size
    #    when swapping one element. This is O(n) per swap check if done carefully, or O(n^2) 
    #    but with much lower constant factors than set comprehension.
    #    Actually, for n=20, O(n^2) is 400 ops. 
    #    Incremental update:
    #    When removing x and adding y:
    #    New Sumset = (Old Sumset - {s | s involves x}) + {s | s involves y}
    #    We need to track counts of each sum to know when to remove from the set.
    #    This requires maintaining a dictionary of sum counts.
    #    Let's implement a class or closure to handle this state.

    class FastSetEvaluator:
        def __init__(self, A):
            self.A = sorted(A)
            self.n = len(self.A)
            self.sum_counts = {}
            self.prod_counts = {}
            self._update()
            
        def _update(self):
            self.sum_counts = {}
            self.prod_counts = {}
            A = self.A
            n = self.n
            for i in range(n):
                ai = A[i]
                for j in range(i, n):
                    aj = A[j]
                    s = ai + aj
                    p = ai * aj
                    self.sum_counts[s] = self.sum_counts.get(s, 0) + 1
                    self.prod_counts[p] = self.prod_counts.get(p, 0) + 1
            
        def score(self):
            return max(len(self.sum_counts), len(self.prod_counts))
        
        def try_swap(self, idx_remove, val_add):
            """
            Returns the new score if we swap A[idx_remove] with val_add.
            If val_add is already in A, returns None.
            """
            if val_add in self.A:
                return None
            
            old_val = self.A[idx_remove]
            if old_val == val_add:
                return self.score()
            
            # Temporarily update counts
            # Remove contributions of old_val
            A = self.A
            n = self.n
            for i in range(n):
                ai = A[i]
                s = ai + old_val
                p = ai * old_val
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    del self.prod_counts[p]
            
            # Add contributions of val_add
            for i in range(n):
                ai = A[i]
                if ai == old_val:
                    continue # We are swapping, so old_val is gone. 
                    # Wait, the loop above removed old_val's interactions with ALL elements including itself.
                    # Now we add val_add's interactions with ALL remaining elements.
                    # Since old_val is removed, we iterate over the remaining n-1 elements.
                    # But my loop iterates over all n elements of A. 
                    # If ai == old_val, we should skip adding interactions with old_val?
                    # No, old_val is NOT in the set anymore. 
                    # So we should only add interactions between val_add and the OTHER elements.
                    # The set A still contains old_val at this point in memory.
                    # So if ai == old_val, we skip.
                    pass
                
                s = ai + val_add
                p = ai * val_add
                self.sum_counts[s] = self.sum_counts.get(s, 0) + 1
                self.prod_counts[p] = self.prod_counts.get(p, 0) + 1
                
            # Calculate new score
            new_score = max(len(self.sum_counts), len(self.prod_counts))
            
            # Revert changes
            # Remove contributions of val_add
            for i in range(n):
                ai = A[i]
                if ai == old_val:
                    continue
                s = ai + val_add
                p = ai * val_add
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    del self.prod_counts[p]
            
            # Add back contributions of old_val
            for i in range(n):
                ai = A[i]
                s = ai + old_val
                p = ai * old_val
                self.sum_counts[s] = self.sum_counts.get(s, 0) + 1
                self.prod_counts[p] = self.prod_counts.get(p, 0) + 1
                
            return new_score

    # Generate candidate pool
    # We want numbers that are "good" for both sums and products.
    # Smooth numbers are good for products.
    # Small numbers are good for sums (compact range).
    
    pool = set()
    # 1 to 50
    for i in range(1, 51):
        pool.add(i)
    # Smooth numbers up to 1000
    for i in range(1, 1001):
        x = i
        is_smooth = True
        # Check if only divisible by 2, 3, 5, 7
        for p in [2, 3, 5, 7]:
            while x % p == 0:
                x //= p
        if x == 1:
            pool.add(i)
    # Add some random numbers up to 2000 to allow exploration
    for _ in range(200):
        pool.add(random.randint(1, 2000))
        
    pool = sorted(pool)
    
    # Initial Candidates
    candidates = []
    
    # 1. Geometric Progression
    # 1, 2, 4, 8, ...
    gp = [2**i for i in range(20)]
    if max(gp) <= 10000:
        candidates.append(gp)
    # 1, 3, 9, ...
    gp3 = [3**i for i in range(10)]
    if len(gp3) >= 20:
        candidates.append(sorted(gp3[:20]))
        
    # 2. Smooth Numbers
    smooths = [x for x in pool if x <= 100]
    if len(smooths) >= 20:
        candidates.append(smooths[:20])
        
    # 3. Mixed: First 10 smooths, next 10 from geometric
    mixed = sorted(list(set(smooths[:10] + [2**i for i in range(10)])))
    if len(mixed) >= 20:
        candidates.append(mixed[:20])
        
    # 4. Random from pool
    for _ in range(5):
        rand_set = random.sample(pool, 20)
        candidates.append(sorted(rand_set))
        
    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    
    for c in candidates:
        if len(c) != 20 or len(set(c)) != 20:
            continue
        if max(c) > 10000:
            continue
        s = calc_score(c)
        if s < best_score:
            best_score = s
            best_set = c[:]
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = calc_score(best_set)
        
    # Local Search
    A = best_set[:]
    current_score = best_score
    evaluator = FastSetEvaluator(A)
    
    T = 5.0
    T_min = 0.5
    cooling = 0.999
    
    # Pre-compute a list of potential replacement values
    # We want to try values from the pool that are not in A
    # To speed up, we can pre-filter the pool to only values <= 2000
    # and shuffle it.
    search_pool = [x for x in pool if x <= 2000]
    random.shuffle(search_pool)
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick index to remove
        idx = random.randint(0, 19)
        
        # Pick new value
        # Bias towards small values? 
        # Let's just pick random from search_pool
        # If search_pool is exhausted or all in A, we need to handle that.
        # Since A has 20 elements and search_pool has ~500, collision is low.
        
        new_val = search_pool[random.randint(0, len(search_pool) - 1)]
        
        if new_val in A:
            continue
            
        new_score = evaluator.try_swap(idx, new_val)
        if new_score is None:
            continue
            
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            A[idx] = new_val
            A.sort()
            current_score = new_score
            evaluator._update() # Rebuild counts for next iteration
            
            if current_score < best_score:
                best_score = current_score
                best_set = A[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            if no_improve_count > max_no_improve:
                # Reheat
                T = 5.0
                no_improve_count = 0
                # Optional: Perturb the set
                # Pick 2 random elements and swap with random pool elements
                # This helps escape local minima
                
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print(calc_score(res))
