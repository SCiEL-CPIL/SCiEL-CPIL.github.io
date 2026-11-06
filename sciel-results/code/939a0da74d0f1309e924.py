import time
import random
import math
import numpy as np

BASE = 51975

# Precompute divisors
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

DIVISORS = get_divisors(BASE)
# Filter to odd divisors (all are odd since BASE is odd)
# Ensure distinct and > 1
DIVISORS = list(set(d for d in DIVISORS if d > 1))
DIVISORS.sort()

class CoveringSystem:
    def __init__(self):
        self.coverage_count = np.zeros(BASE, dtype=np.int32)
        self.uncovered_count = BASE
        self.active = {}  # modulus -> residue
        self.inactive = set(DIVISORS)
        
    def add_modulus(self, m, r):
        """Add modulus m with residue r. Returns change in uncovered_count."""
        if m in self.active:
            return 0
        self.active[m] = r
        self.inactive.discard(m)
        
        delta = 0
        for x in range(r, BASE, m):
            if self.coverage_count[x] == 0:
                delta -= 1
            self.coverage_count[x] += 1
        self.uncovered_count += delta
        return delta

    def remove_modulus(self, m):
        """Remove modulus m. Returns change in uncovered_count."""
        if m not in self.active:
            return 0
        r = self.active[m]
        del self.active[m]
        self.inactive.add(m)
        
        delta = 0
        for x in range(r, BASE, m):
            self.coverage_count[x] -= 1
            if self.coverage_count[x] == 0:
                delta += 1
        self.uncovered_count += delta
        return delta

    def change_residue(self, m, new_r):
        """Change residue of active modulus m to new_r. Returns change in uncovered_count."""
        if m not in self.active:
            return 0
        old_r = self.active[m]
        if old_r == new_r:
            return 0
            
        # Uncover old
        delta = 0
        for x in range(old_r, BASE, m):
            self.coverage_count[x] -= 1
            if self.coverage_count[x] == 0:
                delta += 1
                
        # Cover new
        for x in range(new_r, BASE, m):
            if self.coverage_count[x] == 0:
                delta -= 1
            self.coverage_count[x] += 1
            
        self.active[m] = new_r
        self.uncovered_count += delta
        return delta

    def revert_change_residue(self, m, old_r, new_r):
        """Revert a change_residue operation."""
        # This is effectively change_residue(m, old_r)
        return self.change_residue(m, old_r)

    def get_classes(self):
        return [[m, r] for m, r in self.active.items()]

def greedy_initialization():
    sys_obj = CoveringSystem()
    # We want to add moduli that reduce uncovered_count the most
    # Start with empty
    # Greedy: at each step, try all (m, r) for inactive m, pick best
    # This is O(|D| * N * m) which is too slow if done naively for all m, r.
    # Optimization: 
    # 1. Start with small moduli? Or large?
    # Let's try a heuristic: 
    #    Phase 1: Add moduli in increasing order of m, picking best r for each.
    #    Phase 2: Try to add more.
    
    # Actually, a better greedy is:
    # While True:
    #   best_gain = 0
    #   best_m, best_r = None, None
    #   For each inactive m:
    #      For each r in 0..m-1:
    #         Calculate gain if we added (m, r)
    #         This requires simulating the addition.
    #         Cost: O(m) per r, O(m^2) per m.
    #         Total per iteration: Sum(m^2 for m in inactive).
    #         This might be slow.
        
    # Let's use a simpler heuristic first:
    # Add moduli in descending order of m (large first), picking best r.
    # Then add in ascending order, picking best r.
    
    # Let's try: Sort divisors descending.
    divs_desc = sorted(DIVISORS, reverse=True)
    
    for m in divs_desc:
        best_r = 0
        best_gain = -1
        # Calculate gain for each r
        # Gain = number of currently uncovered residues covered by (m, r)
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if sys_obj.coverage_count[x] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            sys_obj.add_modulus(m, best_r)
        # If gain is 0, adding it doesn't help cover new things, but might be useful later?
        # For now, skip if no immediate gain.
        
    # Now try to add more, maybe smaller ones were skipped?
    # Let's do a second pass ascending
    divs_asc = sorted(DIVISORS)
    for m in divs_asc:
        if m in sys_obj.active:
            continue
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if sys_obj.coverage_count[x] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            sys_obj.add_modulus(m, best_r)
            
    return sys_obj

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try changing residues
        for m in list(sys_obj.active.keys()):
            old_r = sys_obj.active[m]
            # Try all residues? Or random?
            # If m is small, try all. If large, try random.
            if m < 100:
                candidates = range(m)
            else:
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
                # Revert
                sys_obj.change_residue(m, old_r)
                
            if best_new_r != old_r and best_new_score < best_score:
                sys_obj.change_residue(m, best_new_r)
                best_score = best_new_score
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                
        # 2. Try removing moduli
        for m in list(sys_obj.active.keys()):
            r = sys_obj.active[m]
            delta = sys_obj.remove_modulus(m)
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
            else:
                # Revert
                sys_obj.add_modulus(m, r)
                
        # 3. Try adding moduli
        if sys_obj.inactive:
            # Pick a few random inactive moduli
            candidates = random.sample(list(sys_obj.inactive), min(10, len(sys_obj.inactive)))
            for m in candidates:
                best_r = 0
                best_gain = -1
                # Try a few residues
                if m < 50:
                    res_candidates = range(m)
                else:
                    res_candidates = [random.randint(0, m-1) for _ in range(10)]
                
                for r in res_candidates:
                    # Simulate add
                    gain = 0
                    for x in range(r, BASE, m):
                        if sys_obj.coverage_count[x] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                        
                if best_gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        # Revert
                        sys_obj.remove_modulus(m)
                        
        if not improved:
            break
            
    return sys_obj, best_score

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization
    sys_obj = greedy_initialization()
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    sys_obj, best_score = local_search(sys_obj, time_limit=min(10.0, DEADLINE - (time.time() - start_time)))
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 3: Simulated Annealing
    current_score = sys_obj.uncovered_count
    best_score_sa = current_score
    best_sys = sys_obj
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We need a copy of the system to work on
    # Since CoveringSystem doesn't have a deep copy easily, we'll just work on the current one
    # and keep track of the best state manually or restart.
    # Given the complexity, let's just run SA on the current object.
    
    # To allow reverting in SA, we need to be able to undo moves.
    # Our change_residue and add/remove are reversible by calling the inverse.
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            T = T_min
            
        # Pick a move
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
                # Accept
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    # We don't copy the whole system, just note the score
                    # We need to save the state to return it later.
                    # This is a problem. We need to save the state.
                    # Let's save a snapshot of active dict.
                    pass # We'll handle state saving differently
            else:
                # Revert
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
            else:
                sys_obj.add_modulus(m, r)
                
        T *= cooling_rate
        
    # The problem is we didn't save the best state.
    # Let's refine: Keep a `best_active` dict.
    # If we accept a move that improves best_score, update `best_active`.
    
    # Actually, the above loop modifies `sys_obj` in place.
    # If we revert, it goes back.
    # If we accept, it stays.
    # But we need to return the *best* state found, not the final state.
    
    # Let's rewrite the SA loop to track best state.
    
    return sys_obj.get_classes()

# Refining the SA part to properly track best state
def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization
    sys_obj = greedy_initialization()
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    sys_obj, best_score = local_search(sys_obj, time_limit=min(10.0, DEADLINE - (time.time() - start_time)))
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 3: Simulated Annealing
    current_score = sys_obj.uncovered_count
    best_score_sa = current_score
    best_active = sys_obj.active.copy()
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            T = T_min
            
        # Pick a move
        move_type = random.choice(['change', 'add', 'remove'])
        
        old_score = sys_obj.uncovered_count
        accepted = False
        
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
                accepted = True
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
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
                accepted = True
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
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
                accepted = True
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_active = sys_obj.active.copy()
            else:
                sys_obj.add_modulus(m, r)
                
        T *= cooling_rate
        
    # Return the best state found
    return [[m, r] for m, r in best_active.items()]

# We need to make sure the `sys_obj` used in the final return corresponds to `best_active`.
# The `construct` function returns `best_active` which is a dict of {m: r}.
# This is correct.

# One issue: `greedy_initialization` and `local_search` modify `sys_obj`.
# The SA loop also modifies `sys_obj`.
# If SA ends in a bad state, we still return `best_active`.
# This is correct.

# Let's double check the `change_residue` logic.
# It updates `sys_obj.active[m]` and `sys_obj.uncovered_count`.
# If we revert, we call `change_residue(m, old_r)`.
# This should restore the state.

# Potential bug: In `change_residue`, we iterate `range(old_r, BASE, m)` and `range(new_r, BASE, m)`.
# This is correct.

# Let's add a final local search on the best state found by SA?
# The SA might end in a state that is not locally optimal.
# We can reconstruct a CoveringSystem from `best_active` and run a quick local search.

def reconstruct_sys(active_dict):
    sys_obj = CoveringSystem()
    for m, r in active_dict.items():
        sys_obj.add_modulus(m, r)
    return sys_obj

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization
    sys_obj = greedy_initialization()
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    sys_obj, best_score = local_search(sys_obj, time_limit=min(10.0, DEADLINE - (time.time() - start_time)))
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 3: Simulated Annealing
    current_score = sys_obj.uncovered_count
    best_score_sa = current_score
    best_active = sys_obj.active.copy()
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < DEADLINE - 5.0: # Leave 5s for final polish
        if T < T_min:
            T = T_min
            
        # Pick a move
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
            else:
                sys_obj.add_modulus(m, r)
                
        T *= cooling_rate
        
    # Phase 4: Final Polish on Best State
    best_sys = reconstruct_sys(best_active)
    if time.time() - start_time < DEADLINE:
        time_left = DEADLINE - time.time()
        best_sys, final_score = local_search(best_sys, time_limit=time_left)
        if final_score < best_score_sa:
            best_active = best_sys.active.copy()
            
    return [[m, r] for m, r in best_active.items()]
