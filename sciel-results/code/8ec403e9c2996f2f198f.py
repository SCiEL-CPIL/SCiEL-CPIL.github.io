"""Genetic algorithm approach for 3-AP maximization with no 4-AP constraint."""
import numpy as np
import random
import time

# Precompute all 3-APs and 4-APs in [1, 256]
N = 256

def precompute_progressions():
    """Precompute all 3-APs and 4-APs as index triples/quadruples."""
    three_aps = []
    for a in range(N):
        for d in range(1, N - a):
            c = a + 2 * d
            if c < N:
                three_aps.append((a, a + d, c))
    
    four_aps = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c < N:
                four_aps.append((a, a + d, a + 2 * d, c))
                
    return three_aps, four_aps

THREE_APS, FOUR_APS = precompute_progressions()

# Build index maps for fast incremental updates
# three_ap_by_elem[i] = list of indices in THREE_APS that contain i
# four_ap_by_elem[i] = list of indices in FOUR_APS that contain i
three_ap_by_elem = [[] for _ in range(N)]
for idx, (a, b, c) in enumerate(THREE_APS):
    three_ap_by_elem[a].append(idx)
    three_ap_by_elem[b].append(idx)
    three_ap_by_elem[c].append(idx)

four_ap_by_elem = [[] for _ in range(N)]
for idx, (a, b, c, d) in enumerate(FOUR_APS):
    four_ap_by_elem[a].append(idx)
    four_ap_by_elem[b].append(idx)
    four_ap_by_elem[c].append(idx)
    four_ap_by_elem[d].append(idx)

class Individual:
    def __init__(self, mask=0):
        self.mask = mask
        self.three_count = 0
        self.four_count = 0
        # If mask is not 0, we need to calculate counts
        if mask != 0:
            self._recalculate()

    def _recalculate(self):
        """Slow full recalculation for initialization or verification."""
        self.three_count = 0
        self.four_count = 0
        for idx, (a, b, c) in enumerate(THREE_APS):
            if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> c) & 1:
                self.three_count += 1
        for idx, (a, b, c, d) in enumerate(FOUR_APS):
            if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                self.four_count += 1

    def add(self, x):
        """Add element x, updating counts incrementally."""
        if (self.mask >> x) & 1:
            return
        self.mask |= (1 << x)
        
        # Update 3-AP counts
        for idx in three_ap_by_elem[x]:
            a, b, c = THREE_APS[idx]
            # Check if the other two are present
            # x is one of a,b,c. The other two are the ones != x
            if x == a:
                if (self.mask >> b) & 1 and (self.mask >> c) & 1:
                    self.three_count += 1
            elif x == b:
                if (self.mask >> a) & 1 and (self.mask >> c) & 1:
                    self.three_count += 1
            else: # x == c
                if (self.mask >> a) & 1 and (self.mask >> b) & 1:
                    self.three_count += 1

        # Update 4-AP counts
        for idx in four_ap_by_elem[x]:
            a, b, c, d = FOUR_APS[idx]
            if x == a:
                if (self.mask >> b) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                    self.four_count += 1
            elif x == b:
                if (self.mask >> a) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                    self.four_count += 1
            elif x == c:
                if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> d) & 1:
                    self.four_count += 1
            else: # x == d
                if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> c) & 1:
                    self.four_count += 1

    def remove(self, x):
        """Remove element x, updating counts incrementally."""
        if not ((self.mask >> x) & 1):
            return
        self.mask &= ~(1 << x)
        
        # Update 3-AP counts
        for idx in three_ap_by_elem[x]:
            a, b, c = THREE_APS[idx]
            if x == a:
                if (self.mask >> b) & 1 and (self.mask >> c) & 1:
                    self.three_count -= 1
            elif x == b:
                if (self.mask >> a) & 1 and (self.mask >> c) & 1:
                    self.three_count -= 1
            else:
                if (self.mask >> a) & 1 and (self.mask >> b) & 1:
                    self.three_count -= 1

        # Update 4-AP counts
        for idx in four_ap_by_elem[x]:
            a, b, c, d = FOUR_APS[idx]
            if x == a:
                if (self.mask >> b) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                    self.four_count -= 1
            elif x == b:
                if (self.mask >> a) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                    self.four_count -= 1
            elif x == c:
                if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> d) & 1:
                    self.four_count -= 1
            else:
                if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> c) & 1:
                    self.four_count -= 1

    def is_valid(self):
        return self.four_count == 0

    def fitness(self):
        if not self.is_valid():
            return -10**9
        return self.three_count

    def repair(self):
        """Remove elements to fix 4-APs."""
        while self.four_count > 0:
            # Find a 4-AP
            found = False
            for idx, (a, b, c, d) in enumerate(FOUR_APS):
                if (self.mask >> a) & 1 and (self.mask >> b) & 1 and (self.mask >> c) & 1 and (self.mask >> d) & 1:
                    # Remove a random element from this AP
                    candidates = [a, b, c, d]
                    # Heuristic: remove the one that is part of the most 3-APs? 
                    # Or just random to keep GA diversity.
                    # Let's try to remove the one with highest "cost" (most 3-APs involving it)
                    # Actually, just removing one is enough to break this specific 4-AP.
                    # To minimize damage, we could pick the one that appears in the fewest 3-APs?
                    # For speed, random is fine, or we can pick the first one.
                    # Let's pick the one that minimizes loss of 3-APs.
                    min_loss = -1
                    best_rem = None
                    for x in candidates:
                        # Count 3-APs involving x that are currently valid
                        # This is expensive. Just random for now.
                        pass
                    
                    rem = random.choice(candidates)
                    self.remove(rem)
                    found = True
                    break
            if not found:
                break # Should not happen if four_count > 0

    def local_search(self, max_iter=1000):
        """Hill climbing: try to add elements to increase 3-APs while staying valid."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            best_gain = 0
            best_add = -1
            
            # Try adding any element not in mask
            for x in range(N):
                if (self.mask >> x) & 1:
                    continue
                
                # Simulate adding x
                old_three = self.three_count
                old_four = self.four_count
                
                self.add(x)
                
                if self.four_count == 0: # Still valid
                    gain = self.three_count - old_three
                    if gain > best_gain:
                        best_gain = gain
                        best_add = x
                
                # Revert
                self.remove(x)
                
            if best_add != -1:
                self.add(best_add)
                improved = True
                
        return self

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    pop_size = 50
    generations = 50
    mutation_rate = 0.1
    
    # Initialize population
    population = []
    
    # 1. Greedy Initial Solution
    greedy_ind = Individual(0)
    # Greedy add: sort by some heuristic? Just add 1..256 if valid
    for i in range(N):
        greedy_ind.add(i)
        if not greedy_ind.is_valid():
            greedy_ind.remove(i)
    greedy_ind.local_search()
    population.append(greedy_ind)
    
    # 2. Random Initial Solutions
    for _ in range(pop_size - 1):
        ind = Individual(0)
        # Random subset
        for i in range(N):
            if random.random() < 0.5:
                ind.add(i)
        ind.repair()
        ind.local_search()
        population.append(ind)
    
    best_ind = max(population, key=lambda x: x.fitness())
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_ind)
        
        while len(new_population) < pop_size:
            # Selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            # Uniform crossover
            mask1 = p1.mask
            mask2 = p2.mask
            child_mask = 0
            for i in range(N):
                if random.random() < 0.5:
                    if (mask1 >> i) & 1:
                        child_mask |= (1 << i)
                else:
                    if (mask2 >> i) & 1:
                        child_mask |= (1 << i)
            
            child = Individual(child_mask)
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    if (child.mask >> i) & 1:
                        child.remove(i)
                    else:
                        child.add(i)
            
            # Repair
            child.repair()
            
            # Local Search
            child.local_search()
            
            new_population.append(child)
        
        population = new_population
        current_best = max(population, key=lambda x: x.fitness())
        if current_best.fitness() > best_ind.fitness():
            best_ind = current_best
            # Optional: print progress
            # print(f"Gen {gen}: Best Fitness {best_ind.fitness()}")
    
    # Convert best_ind.mask to list
    result = [i + 1 for i in range(N) if (best_ind.mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# Pre-run to cache result
_CACHED_RESULT = None

def _ensure_cached():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        _CACHED_RESULT = construct()
    return _CACHED_RESULT
