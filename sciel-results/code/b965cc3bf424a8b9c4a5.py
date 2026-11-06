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

    def get_classes(self):
        return [[m, r] for m, r in self.active.items()]

def find_best_residue(m, coverage_count):
    """
    Finds the residue r in [0, m) that covers the most currently uncovered residues
    given the current coverage_count (assuming m is NOT active in coverage_count).
    Returns (best_r, max_gain).
    """
    gains = np.zeros(m, dtype=np.int32)
    indices = np.arange(BASE)
    uncovered_mask = coverage_count == 0
    if not np.any(uncovered_mask):
        return 0, 0
    residues = indices[uncovered_mask] % m
    gains = np.bincount(residues, minlength=m)
    best_r = int(np.argmax(gains))
    max_gain = int(gains[best_r])
    return best_r, max_gain

def greedy_initialization():
    """Try both ascending and descending greedy, return the better one."""
    def _greedy(order):
        sys_obj = CoveringSystem()
        for m in order:
            best_r, gain = find_best_residue(m, sys_obj.coverage_count)
            if gain > 0:
                sys_obj.add_modulus(m, best_r)
        # Second pass: try adding any remaining
        for m in order:
            if m in sys_obj.active:
                continue
            best_r, gain = find_best_residue(m, sys_obj.coverage_count)
            if gain > 0:
                sys_obj.add_modulus(m, best_r)
        return sys_obj

    divs_asc = sorted(DIVISORS)
    divs_desc = sorted(DIVISORS, reverse=True)

    sys_asc = _greedy(divs_asc)
    sys_desc = _greedy(divs_desc)

    if sys_desc.uncovered_count < sys_asc.uncovered_count:
        return sys_desc
    return sys_asc

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try changing residues using exact best improvement
        for m in list(sys_obj.active.keys()):
            old_r = sys_obj.active[m]
            
            # Temporarily remove m to see the landscape
            delta_rem = sys_obj.remove_modulus(m)
            # 2. Find best residue for m in the new landscape
            best_new_r, gain = find_best_residue(m, sys_obj.coverage_count)
            # 3. Add m with best_new_r
            delta_add = sys_obj.add_modulus(m, best_new_r)
            
            new_score = sys_obj.uncovered_count
            
            if new_score < best_score:
                best_score = new_score
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
            else:
                # Revert: Remove m, Add m with old_r
                sys_obj.remove_modulus(m)
                sys_obj.add_modulus(m, old_r)
                
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
            for m in list(sys_obj.inactive):
                best_r, gain = find_best_residue(m, sys_obj.coverage_count)
                if gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        # Revert
                        sys_obj.remove_modulus(m)
                        
        # 4. Try 2-opt: change two residues simultaneously
        active_keys = list(sys_obj.active.keys())
        if len(active_keys) >= 2 and time.time() - start_time < time_limit - 0.5:
            # Sample pairs to keep it fast
            max_pairs = min(50, len(active_keys) * (len(active_keys) - 1) // 2)
            random.shuffle(active_keys)
            pairs_tried = 0
            for i in range(len(active_keys)):
                for j in range(i+1, len(active_keys)):
                    if pairs_tried >= max_pairs:
                        break
                    m1 = active_keys[i]
                    m2 = active_keys[j]
                    r1_old = sys_obj.active[m1]
                    r2_old = sys_obj.active[m2]
                    
                    # Remove both
                    sys_obj.remove_modulus(m1)
                    sys_obj.remove_modulus(m2)
                    
                    # Find best residues for both
                    best_r1, gain1 = find_best_residue(m1, sys_obj.coverage_count)
                    # Add m1 temporarily to update coverage for m2's search
                    sys_obj.add_modulus(m1, best_r1)
                    best_r2, gain2 = find_best_residue(m2, sys_obj.coverage_count)
                    sys_obj.add_modulus(m2, best_r2)
                    
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        # Revert
                        sys_obj.remove_modulus(m1)
                        sys_obj.remove_modulus(m2)
                        sys_obj.add_modulus(m1, r1_old)
                        sys_obj.add_modulus(m2, r2_old)
                    
                    pairs_tried += 1
                    if time.time() - start_time >= time_limit:
                        break
                
        if not improved:
            break
            
    return sys_obj, best_score

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    # Phase 1: Greedy Initialization (tries both orders)
    sys_obj = greedy_initialization()

    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()

    # Phase 2: Local Search
    sys_obj, best_score = local_search(sys_obj, time_limit=min(20.0, DEADLINE - (time.time() - start_time)))

    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()

    # Phase 3: Multi-start with targeted perturbations
    best_global_score = best_score
    best_global_active = sys_obj.active.copy()
    
    # Try multiple perturbation strategies
    num_starts = 0
    max_starts = 8
    
    while time.time() - start_time < DEADLINE - 5.0 and num_starts < max_starts:
        # Copy current best
        sys_obj2 = CoveringSystem()
        for m, r in best_global_active.items():
            sys_obj2.add_modulus(m, r)

        # Strategy A: Random perturbation of a few residues
        if sys_obj2.active and random.random() < 0.5:
            keys = list(sys_obj2.active.keys())
            n_perturb = random.randint(2, min(4, len(keys)))
            for _ in range(n_perturb):
                m = random.choice(keys)
                new_r = random.randint(0, m - 1)
                sys_obj2.change_residue(m, new_r)
        
        # Strategy B: Remove one modulus and re-add with best residue
        elif sys_obj2.active and random.random() < 0.5:
            keys = list(sys_obj2.active.keys())
            m_remove = random.choice(keys)
            sys_obj2.remove_modulus(m_remove)
            best_r, gain = find_best_residue(m_remove, sys_obj2.coverage_count)
            if gain > 0:
                sys_obj2.add_modulus(m_remove, best_r)
            else:
                # Try a random residue if best gain is 0
                sys_obj2.add_modulus(m_remove, random.randint(0, m_remove - 1))
        
        # Strategy C: Swap residues between two moduli (if same modulus not required)
        # Actually moduli must be distinct, so we can't swap moduli.
        # Instead: change residue of the modulus with smallest contribution
        else:
            if sys_obj2.active:
                # Find modulus whose removal causes least damage (smallest unique coverage)
                keys = list(sys_obj2.active.keys())
                if len(keys) >= 2:
                    # Pick two random moduli and swap their residues if possible
                    m1, m2 = random.sample(keys, 2)
                    r1 = sys_obj2.active[m1]
                    r2 = sys_obj2.active[m2]
                    # Only swap if residues are valid for both moduli
                    if r2 < m1 and r1 < m2:
                        sys_obj2.change_residue(m1, r2)
                        sys_obj2.change_residue(m2, r1)

        # Run local search on perturbed state
        remaining_time = DEADLINE - 5.0 - (time.time() - start_time)
        if remaining_time > 1.0:
            sys_obj2, score2 = local_search(sys_obj2, time_limit=remaining_time)
            if score2 < best_global_score:
                best_global_score = score2
                best_global_active = sys_obj2.active.copy()
        
        num_starts += 1

    return [[m, r] for m, r in best_global_active.items()]
