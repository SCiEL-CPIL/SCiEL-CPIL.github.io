import time
import random
import math
import numpy as np

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i > 1:
                divs.append(i)
            other = n // i
            if other > 1 and other != i:
                divs.append(other)
    return divs

DIVISORS = sorted(set(d for d in get_divisors(BASE) if d > 1))

class CoveringSystem:
    def __init__(self):
        self.coverage_count = np.zeros(BASE, dtype=np.int32)
        self.uncovered_count = BASE
        self.active = {}  # modulus -> residue
        self.inactive = set(DIVISORS)
        
    def add_modulus(self, m, r):
        if m in self.active:
            return 0
        self.active[m] = r
        self.inactive.discard(m)
        
        indices = np.arange(r, BASE, m)
        # Count how many were 0
        new_uncovered_added = np.sum(self.coverage_count[indices] == 0)
        self.coverage_count[indices] += 1
        self.uncovered_count -= new_uncovered_added
        return -new_uncovered_added

    def remove_modulus(self, m):
        if m not in self.active:
            return 0
        r = self.active[m]
        del self.active[m]
        self.inactive.add(m)
        
        indices = np.arange(r, BASE, m)
        # Count how many become 0
        new_uncovered_removed = np.sum(self.coverage_count[indices] == 1)
        self.coverage_count[indices] -= 1
        self.uncovered_count += new_uncovered_removed
        return new_uncovered_removed

    def change_residue(self, m, new_r):
        if m not in self.active:
            return 0
        old_r = self.active[m]
        if old_r == new_r:
            return 0
            
        old_indices = np.arange(old_r, BASE, m)
        new_indices = np.arange(new_r, BASE, m)
        
        # Uncover old
        # Count how many become 0
        new_uncovered_from_old = np.sum(self.coverage_count[old_indices] == 1)
        self.coverage_count[old_indices] -= 1
        
        # Cover new
        # Count how many were 0
        new_covered_from_new = np.sum(self.coverage_count[new_indices] == 0)
        self.coverage_count[new_indices] += 1
        
        self.active[m] = new_r
        delta = new_uncovered_from_old - new_covered_from_new
        self.uncovered_count += delta
        return delta

    def get_classes(self):
        return [[m, r] for m, r in self.active.items()]

def greedy_initialization():
    sys_obj = CoveringSystem()
    # Try descending and ascending, pick best
    best_sys = None
    best_score = BASE
    
    for order in [sorted(DIVISORS, reverse=True), sorted(DIVISORS)]:
        temp_sys = CoveringSystem()
        for m in order:
            # Find best r
            # We can check all r in 0..m-1
            # Gain = sum(coverage_count[arange(r, BASE, m)] == 0)
            # This is O(m * (BASE/m)) = O(BASE) per m if done naively?
            # No, arange is O(BASE/m). Sum is O(BASE/m). Total O(BASE).
            # For each m, we do this for all r. Total O(m * BASE) = O(BASE * m).
            # Sum over m is O(BASE * sum(m)). Sum of divisors is roughly O(BASE log log BASE).
            # 51975 * sum(divisors) is manageable.
            
            best_r = 0
            best_gain = -1
            
            # Optimization: If m is large, we can sample r?
            # No, for greedy we want the best.
            # But we can vectorize the gain calculation for all r?
            # Gain(r) = count of x in {r, r+m, ...} such that coverage_count[x] == 0.
            # This is a sum over a strided array.
            
            # Let's just loop r for now, it's fast enough for 51975.
            for r in range(m):
                indices = np.arange(r, BASE, m)
                gain = np.sum(temp_sys.coverage_count[indices] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                temp_sys.add_modulus(m, best_r)
        
        if temp_sys.uncovered_count < best_score:
            best_score = temp_sys.uncovered_count
            best_sys = temp_sys
            
    return best_sys

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    best_active = sys_obj.active.copy()
    best_coverage = sys_obj.coverage_count.copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try changing residues
        # Iterate over active moduli
        for m in list(sys_obj.active.keys()):
            old_r = sys_obj.active[m]
            
            # Determine candidates
            if m <= 20:
                candidates = range(m)
            else:
                # Sample 20 random residues
                candidates = [random.randint(0, m-1) for _ in range(20)]
            
            best_new_r = old_r
            best_new_score = sys_obj.uncovered_count
            
            for r in candidates:
                if r == old_r:
                    continue
                delta = sys_obj.change_residue(m, r)
                new_score = sys_obj.uncovered_count
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_r = r
                sys_obj.change_residue(m, old_r)
                
            if best_new_r != old_r and best_new_score < best_score:
                sys_obj.change_residue(m, best_new_r)
                best_score = best_new_score
                best_active = sys_obj.active.copy()
                best_coverage = sys_obj.coverage_count.copy()
                improved = True

        # 2. Try removing moduli
        for m in list(sys_obj.active.keys()):
            r = sys_obj.active[m]
            delta = sys_obj.remove_modulus(m)
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                best_active = sys_obj.active.copy()
                best_coverage = sys_obj.coverage_count.copy()
                improved = True
            else:
                sys_obj.add_modulus(m, r)
                
        # 3. Try adding moduli
        if sys_obj.inactive:
            # Pick a few random inactive moduli
            candidates = random.sample(list(sys_obj.inactive), min(10, len(sys_obj.inactive)))
            for m in candidates:
                # Find best r for this m
                if m <= 20:
                    res_candidates = range(m)
                else:
                    res_candidates = [random.randint(0, m-1) for _ in range(20)]
                
                best_r = 0
                best_gain = -1
                for r in res_candidates:
                    # Calculate gain without modifying state
                    # Gain = number of uncovered residues in this class
                    indices = np.arange(r, BASE, m)
                    gain = np.sum(sys_obj.coverage_count[indices] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                        
                if best_gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        best_active = sys_obj.active.copy()
                        best_coverage = sys_obj.coverage_count.copy()
                        improved = True
                    else:
                        sys_obj.remove_modulus(m)
                        
        if not improved:
            # Restart from best state?
            # Or just break?
            # Let's restart to escape local minima
            sys_obj.active = best_active.copy()
            sys_obj.coverage_count = best_coverage.copy()
            sys_obj.uncovered_count = best_score
            # Recalculate inactive
            sys_obj.inactive = set(DIVISORS) - set(best_active.keys())
            
            # If we restart too many times, break
            # Simple counter or just rely on time limit
            
    # Restore best state
    sys_obj.active = best_active
    sys_obj.coverage_count = best_coverage
    sys_obj.uncovered_count = best_score
    sys_obj.inactive = set(DIVISORS) - set(best_active.keys())
    
    return sys_obj, best_score

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization
    sys_obj = greedy_initialization()
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    time_left = DEADLINE - (time.time() - start_time)
    # Use most of the time for local search with restarts
    # We'll split time: 80% local search, 20% SA
    ls_time = time_left * 0.8
    sa_time = time_left * 0.2
    
    sys_obj, best_score = local_search(sys_obj, time_limit=ls_time)
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 3: Simulated Annealing
    current_score = sys_obj.uncovered_count
    best_score_sa = current_score
    best_active = sys_obj.active.copy()
    best_coverage = sys_obj.coverage_count.copy()
    
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Run SA for remaining time
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            T = T_min
            
        move_type = random.choice(['change', 'add', 'remove'])
        
        old_score = sys_obj.uncovered_count
        
        if move_type == 'change':
            if not sys_obj.active:
                continue
            m = random.choice(list(sys_obj.active.keys()))
            old_r = sys_obj.active[m]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            sys_obj.change_residue(m, new_r)
            new_score = sys_obj.uncovered_count
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
                    best_coverage = sys_obj.coverage_count.copy()
            else:
                sys_obj.change_residue(m, old_r)
                
        elif move_type == 'add':
            if not sys_obj.inactive:
                continue
            m = random.choice(list(sys_obj.inactive))
            r = random.randint(0, m - 1)
            sys_obj.add_modulus(m, r)
            new_score = sys_obj.uncovered_count
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
                    best_coverage = sys_obj.coverage_count.copy()
            else:
                sys_obj.remove_modulus(m)
                
        elif move_type == 'remove':
            if not sys_obj.active:
                continue
            m = random.choice(list(sys_obj.active.keys()))
            r = sys_obj.active[m]
            sys_obj.remove_modulus(m)
            new_score = sys_obj.uncovered_count
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
                    best_coverage = sys_obj.coverage_count.copy()
            else:
                sys_obj.add_modulus(m, r)
                
        T *= cooling_rate
        
    # Return best state
    return [[m, r] for m, r in best_active.items()]
