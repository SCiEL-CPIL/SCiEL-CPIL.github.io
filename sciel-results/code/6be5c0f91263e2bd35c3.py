import time
import random
import numpy as np

BASE = 51975

# Generate all divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute positions for each divisor and residue
# _POS[i][r] is an array of indices x in [0, BASE) such that x % _DIVS[i] == r
_POS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        lst.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(lst)

# Map from modulus value to index
_DIV_IDX = {m: i for i, m in enumerate(_DIVS)}

def _score_from_counts(counts):
    return int(np.count_nonzero(counts == 0))

def _build_counts(residues):
    """
    residues: list of length N_DIVS, where residues[i] is the residue for _DIVS[i]
    """
    counts = np.zeros(BASE, dtype=np.int16)
    for i, m in enumerate(_DIVS):
        r = residues[i]
        counts[_POS[i][r]] += 1
    return counts

def _greedy_init(rng):
    """
    Initialize residues using a greedy strategy.
    Order: Random, but biased towards smaller moduli (more coverage) or larger?
    Smaller moduli cover more points, so they have more impact.
    Let's try: Pick a random order of divisors. For each, pick the residue that covers most currently uncovered points.
    """
    counts = np.zeros(BASE, dtype=np.int16)
    residues = [0] * N_DIVS
    
    # Heuristic: Process smaller moduli first? Or random?
    # Let's try random order but maybe bias towards small moduli in early attempts?
    # Actually, let's just do random order for diversity, or sorted by size.
    # Small moduli (e.g., 3) cover 1/3 of the space. Large (e.g., 51975/3=17325? No, max divisor is 17325? No, max is 51975/1? No, m>1. Max m is 51975/3? No, 51975 is a divisor? No, m must divide 51975. Max m is 51975.
    # If m=51975, it covers 1 point.
    # If m=3, it covers 17325 points.
    
    # Strategy: Assign residues to small moduli first to "block" the space, then fill gaps with larger moduli?
    # Or assign large moduli first to pin down specific points?
    # Let's try sorting by modulus size (ascending).
    order = list(range(N_DIVS))
    # _DIVS is already sorted ascending.
    
    for i in order:
        m = _DIVS[i]
        best_r = 0
        best_gain = -1
        for r in range(m):
            positions = _POS[i][r]
            # Count how many of these positions are currently uncovered (count == 0)
            gain = int(np.count_nonzero(counts[positions] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        counts[_POS[i][best_r]] += 1
        
    return residues, counts

def _simulated_annealing(residues, counts, deadline, rng):
    """
    Improve the solution using Simulated Annealing.
    """
    current_score = _score_from_counts(counts)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    start_temp = 100.0
    end_temp = 0.1
    # Cooling schedule
    # We don't know the number of steps in advance, so we use time-based cooling or step-based with a large step count.
    # Let's use a fixed number of steps per "phase" or just run until deadline.
    
    t_start = time.time()
    temp = start_temp
    # Estimate steps: ~1000 steps/sec?
    # Let's just decay temp based on elapsed time.
    # T(t) = T0 * (T_end/T0)^(t / T_total)
    T_total = 60.0 # seconds
    
    while time.time() < deadline:
        elapsed = time.time() - t_start
        if elapsed > T_total:
            temp = end_temp
        else:
            temp = start_temp * (end_temp / start_temp) ** (elapsed / T_total)
            
        # Pick a random divisor to change
        i = rng.randint(0, N_DIVS - 1)
        m = _DIVS[i]
        old_r = residues[i]
        
        # Pick a new random residue
        if m == 1:
            new_r = 0
        else:
            new_r = rng.randint(0, m - 1)
            
        if new_r == old_r:
            continue
            
        old_positions = _POS[i][old_r]
        new_positions = _POS[i][new_r]
        
        # Calculate delta score
        # Points lost: those in old_positions that have count == 1 (will become 0)
        # Points gained: those in new_positions that have count == 0 (will become 1)
        
        # Efficient calculation:
        # We need to check counts before modification.
        
        # Count losses
        # positions in old_positions where counts[x] == 1
        # We can do this by:
        # temp_counts = counts[old_positions]
        # losses = np.count_nonzero(temp_counts == 1)
        
        # Count gains
        # positions in new_positions where counts[x] == 0
        # gains = np.count_nonzero(counts[new_positions] == 0)
        
        # However, there is an overlap!
        # If a position is in BOTH old and new, and count was 1, it loses 1 and gains 1?
        # No, the class moves.
        # If x is in old and new:
        #   Before: count[x] included this class.
        #   After: count[x] still includes this class.
        #   So count[x] doesn't change for x in intersection.
        #   So we should only look at (old \ new) for losses and (new \ old) for gains.
        
        # Intersection check is expensive if we do set operations on numpy arrays of size ~N/m.
        # But N/m is small for large m. For small m, N/m is large.
        # Max N/m is 51975/3 = 17325.
        # Intersection of two sets of size 17325.
        
        # Alternative: Just apply the change, compute new score, revert if not accepted.
        # Applying change:
        # counts[old_positions] -= 1
        # counts[new_positions] += 1
        # new_score = _score_from_counts(counts)
        # delta = new_score - current_score
        
        # This is O(N) for score calculation? No, _score_from_counts is O(N).
        # N = 51975.
        # 1000 steps/sec * 51975 ops/step = 50M ops/sec.
        # NumPy is fast, but counting non-zeros on 51975 elements takes ~10-20us.
        # So ~50k steps/sec is possible.
        # In 60s, ~3M steps. This is reasonable.
        
        # Let's optimize the score calculation.
        # We only need to update the score if the change affects the count of zeros.
        # But maintaining a running score is tricky with overlaps.
        # Let's stick to the "Apply, Score, Revert" method first, as it's simpler and robust.
        
        counts[old_positions] -= 1
        counts[new_positions] += 1
        
        new_score = _score_from_counts(counts)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            # Accept
            residues[i] = new_r
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
                
                if best_score == 0:
                    return best_residues, best_counts
        else:
            # Reject: Revert
            counts[new_positions] -= 1
            counts[old_positions] += 1
            
    return best_residues, best_counts

def construct():
    start_time = time.time()
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    best_residues = None
    best_score = BASE + 1
    best_counts = None
    
    # Phase 1: Multiple Greedy Initializations
    # Try different orders
    # 1. Ascending moduli
    # 2. Descending moduli
    # 3. Random order
    
    orders = [
        list(range(N_DIVS)), # Ascending
        list(range(N_DIVS - 1, -1, -1)), # Descending
    ]
    
    # Add a few random orders
    for _ in range(5):
        o = list(range(N_DIVS))
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 20.0:
            break
            
        # Modify _greedy_init to accept an order
        counts = np.zeros(BASE, dtype=np.int16)
        residues = [0] * N_DIVS
        
        for i in order:
            m = _DIVS[i]
            best_r = 0
            best_gain = -1
            for r in range(m):
                positions = _POS[i][r]
                gain = int(np.count_nonzero(counts[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[i] = best_r
            counts[_POS[i][best_r]] += 1
            
        s = _score_from_counts(counts)
        
        if s < best_score:
            best_score = s
            best_residues = residues[:]
            best_counts = counts.copy()
            
        # Run a quick local search (SA) on this initialization
        # Budget: 5 seconds per init?
        sa_deadline = min(deadline - 10.0, time.time() + 10.0)
        improved_res, improved_counts = _simulated_annealing(residues, counts, sa_deadline, rng)
        s_improved = _score_from_counts(improved_counts)
        
        if s_improved < best_score:
            best_score = s_improved
            best_residues = improved_res
            best_counts = improved_counts
            
            if best_score == 0:
                # Convert residues to classes
                classes = [[_DIVS[i], best_residues[i]] for i in range(N_DIVS)]
                return classes

    # Phase 2: Final long SA run on the best solution found
    if best_residues is not None:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            final_res, final_counts = _simulated_annealing(best_residues, best_counts, time.time() + remaining, rng)
            s_final = _score_from_counts(final_counts)
            if s_final < best_score:
                best_score = s_final
                best_residues = final_res
                best_counts = final_counts

    # Convert best_residues to required format
    classes = [[_DIVS[i], best_residues[i]] for i in range(N_DIVS)]
    return classes

