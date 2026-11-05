import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    D = 2 * n + 1  # 15 neighbors (including self)

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors
    neighbors = np.zeros((N, D), dtype=np.int32)
    for i in range(N):
        w = idx_to_tuple(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    neighbors[i, count] = tuple_to_idx(new_w)
                    count += 1
        # Ensure length is D (it should be, 1 + 2*7 = 15)
        assert count == D, f"Expected {D} neighbors, got {count}"

    # State for an individual
    # We use numpy arrays for speed
    # code_mask: bool array, 1 if vertex is in code
    # coverage: int array, count of codewords covering each vertex
    
    class Individual:
        __slots__ = ['mask', 'coverage', 'size', 'uncovered', 'fitness']
        
        def __init__(self, mask=None, coverage=None):
            if mask is None:
                self.mask = np.zeros(N, dtype=bool)
                self.coverage = np.zeros(N, dtype=np.int32)
                self.size = 0
                self.uncovered = N
                self.fitness = N
            else:
                self.mask = mask.copy()
                self.coverage = coverage.copy()
                self.size = np.sum(self.mask)
                self.uncovered = np.sum(self.coverage == 0)
                self.fitness = self.size + 10 * self.uncovered # Large penalty for uncovered
            
        def clone(self):
            return Individual(self.mask, self.coverage)
        
        def add(self, v):
            if self.mask[v]:
                return 0
            self.mask[v] = True
            self.size += 1
            # Update coverage
            nbs = neighbors[v]
            prev_cov = self.coverage[nbs]
            new_cov = prev_cov + 1
            self.coverage[nbs] = new_cov
            # Update uncovered count
            # Vertices that were uncovered (prev_cov == 0) and are now covered (new_cov > 0)
            # Since we only add 1, prev_cov == 0 is the only case where it was uncovered.
            newly_covered = np.sum(prev_cov == 0)
            self.uncovered -= newly_covered
            self.fitness = self.size + 10 * self.uncovered
            return newly_covered

        def remove(self, v):
            if not self.mask[v]:
                return 0
            self.mask[v] = False
            self.size -= 1
            nbs = neighbors[v]
            prev_cov = self.coverage[nbs]
            new_cov = prev_cov - 1
            self.coverage[nbs] = new_cov
            # Vertices that were covered (prev_cov > 0) and are now uncovered (new_cov == 0)
            newly_uncovered = np.sum(new_cov == 0)
            self.uncovered += newly_uncovered
            self.fitness = self.size + 10 * self.uncovered
            return newly_uncovered

        def get_code_list(self):
            return [idx for idx in range(N) if self.mask[idx]]

    # 1. Greedy Initialization
    def greedy_init():
        ind = Individual()
        # Pick a random start
        start = random.randint(0, N-1)
        ind.add(start)
        
        # Simple greedy: pick vertex with max new coverage
        # To speed up, we can just do a few passes
        # For N=2187, a full scan is ~30k ops. 100 iterations = 3M ops. Fast.
        for _ in range(200):
            if ind.uncovered == 0:
                break
            best_v = -1
            best_gain = 0
            # Sample a subset of candidates to speed up? 
            # Or just scan all. Scanning all is safer for quality.
            for v in range(N):
                if ind.mask[v]:
                    continue
                # Gain is number of neighbors with coverage 0
                gain = np.sum(ind.coverage[neighbors[v]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                if gain == ind.uncovered:
                    break
            if best_v == -1 or best_gain == 0:
                break
            ind.add(best_v)
            
        # Repair if needed (should be covered by greedy, but just in case)
        while ind.uncovered > 0:
            # Find an uncovered vertex
            unc_idx = np.where(ind.coverage == 0)[0]
            if len(unc_idx) == 0: break
            target = unc_idx[0]
            # Find a neighbor of target to add
            # Pick the neighbor that covers the most other uncovered
            best_v = -1
            best_gain = 0
            for v in neighbors[target]:
                if ind.mask[v]: continue
                gain = np.sum(ind.coverage[neighbors[v]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1: break
            ind.add(best_v)
            
        return ind

    # 2. Simulated Annealing Local Search
    def local_search(ind, time_limit):
        T = 1.0
        T_min = 0.001
        cooling = 0.99995 # Very slow cooling to allow many steps
        end_time = time.time() + time_limit
        
        # Precompute random neighbors? No, just pick random v.
        # We need to be careful with the move selection.
        # Moves: Add, Remove, Swap
        
        while time.time() < end_time:
            # Pick a random move type
            r = random.random()
            if r < 0.3:
                # Remove
                # Pick a random vertex from code
                if ind.size == 0: continue
                # Efficient way to pick random True from mask:
                # np.nonzero is slow if called every time. 
                # Maintain a list of indices? 
                # For now, use np.nonzero but cache it? 
                # Actually, for N=2187, np.nonzero is ~microseconds.
                code_indices = np.nonzero(ind.mask)[0]
                if len(code_indices) == 0: continue
                v = code_indices[random.randint(0, len(code_indices)-1)]
                
                # Calculate Delta
                # If we remove v, size -1.
                # Uncovered increases by count of neighbors with coverage 1.
                nbs = neighbors[v]
                newly_unc = np.sum(ind.coverage[nbs] == 1)
                delta_f = -1 + 10 * newly_unc
                
                if delta_f < 0 or random.random() < np.exp(-delta_f / T):
                    ind.remove(v)
            elif r < 0.6:
                # Add
                # Pick random vertex not in code
                # Efficient: pick random v, if in code, try again? 
                # Or use np.nonzero(~mask)
                non_code = np.nonzero(~ind.mask)[0]
                if len(non_code) == 0: continue
                v = non_code[random.randint(0, len(non_code)-1)]
                
                # Calculate Delta
                # Size +1.
                # Uncovered decreases by count of neighbors with coverage 0.
                nbs = neighbors[v]
                newly_cov = np.sum(ind.coverage[nbs] == 0)
                delta_f = 1 - 10 * newly_cov
                
                if delta_f < 0 or random.random() < np.exp(-delta_f / T):
                    ind.add(v)
            else:
                # Swap
                if ind.size == 0: continue
                code_indices = np.nonzero(ind.mask)[0]
                non_code = np.nonzero(~ind.mask)[0]
                if len(non_code) == 0: continue
                
                v_rem = code_indices[random.randint(0, len(code_indices)-1)]
                v_add = non_code[random.randint(0, len(non_code)-1)]
                
                if v_rem == v_add: continue
                
                # Simulate
                # Remove v_rem
                nbs_rem = neighbors[v_rem]
                cov_rem = ind.coverage[nbs_rem]
                # Add v_add
                nbs_add = neighbors[v_add]
                cov_add = ind.coverage[nbs_add]
                
                # This is complex to calculate delta directly without modifying.
                # Let's just do it in two steps if accepted? 
                # Or calculate carefully.
                # Let's skip Swap for simplicity in SA, Add/Remove is sufficient for connectivity.
                # Or implement Swap as Remove then Add.
                pass

            T = max(T_min, T * cooling)
            
            # Occasionally, do a greedy repair step to fix broken coverage?
            # No, SA handles it via penalty.

        # Final Greedy Polish
        # Remove redundant
        changed = True
        while changed and time.time() < end_time:
            changed = False
            code_indices = np.nonzero(ind.mask)[0]
            random.shuffle(code_indices) # Random order to avoid bias
            for v in code_indices:
                if not ind.mask[v]: continue
                # Check if redundant
                # Redundant if all neighbors have coverage > 1
                if np.min(ind.coverage[neighbors[v]]) > 1:
                    ind.remove(v)
                    changed = True
        
        # Fill holes if any
        while ind.uncovered > 0 and time.time() < end_time:
            unc_idx = np.where(ind.coverage == 0)[0]
            if len(unc_idx) == 0: break
            target = unc_idx[0]
            best_v = -1
            best_gain = 0
            for v in neighbors[target]:
                if ind.mask[v]: continue
                gain = np.sum(ind.coverage[neighbors[v]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1: break
            ind.add(best_v)
            
        return ind

    # 3. Genetic Algorithm / Memetic
    population_size = 10
    pop = []
    
    # Initialize population
    for i in range(population_size):
        if i == 0:
            ind = greedy_init()
        else:
            # Perturb greedy
            ind = greedy_init()
            # Random mutations
            for _ in range(50):
                if random.random() < 0.5:
                    v = random.randint(0, N-1)
                    if ind.mask[v]:
                        ind.remove(v)
                    else:
                        ind.add(v)
            # Quick repair
            while ind.uncovered > 0:
                unc_idx = np.where(ind.coverage == 0)[0]
                if len(unc_idx) == 0: break
                target = unc_idx[0]
                for v in neighbors[target]:
                    if not ind.mask[v]:
                        ind.add(v)
                        break
        pop.append(ind)
    
    best_ind = min(pop, key=lambda x: x.fitness)
    
    gen = 0
    while time.time() < deadline - 5:
        new_pop = []
        # Elitism
        new_pop.append(best_ind.clone())
        
        for _ in range(population_size - 1):
            # Selection: Tournament
            p1 = random.choice(pop)
            p2 = random.choice(pop)
            if p1.fitness < p2.fitness:
                parent = p1
            else:
                parent = p2
            
            # Crossover: Take parent, mutate
            child = parent.clone()
            
            # Mutation: SA for a short time
            # Allocate time based on remaining budget
            # Simple: fixed small time or steps
            # Let's use time-based
            time_per_child = (deadline - time.time()) / (population_size * (gen + 1))
            if time_per_child < 0.1:
                time_per_child = 0.1
            
            child = local_search(child, time_per_child)
            new_pop.append(child)
            
            if child.fitness < best_ind.fitness:
                best_ind = child.clone()
        
        pop = new_pop
        gen += 1
        
        # If best is perfect (size 186?), we can stop early? 
        # 186 is best known. If we hit 186, we can try to break it or stop.
        if best_ind.size == 186 and best_ind.uncovered == 0:
             # Try to break 186?
             # Or just stop and return 186.
             # Let's try to improve further.
             pass

    # Final Polish on Best
    best_ind = local_search(best_ind, deadline - time.time() - 1)
    
    result = best_ind.get_code_list()
    # Convert to tuples
    return [idx_to_tuple(c) for c in result]
