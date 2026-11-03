"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time

# EVOLVE-BLOCK-START
def _score_of(U):
    """Return the raw lower-bound score for a concrete set U (must contain 0)."""
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    mx = u[-1]
    if mx <= 0:
        return 0.0
    # difference set U - U
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    # sum set U + U
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build_geometric(b, m):
    """Construct U = {floor(b^k) | k=0..m} as a list containing 0."""
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    return U

def construct():
    deadline = time.time() + 110.0  # Leave some buffer
    best_score = 0.0
    best_set = [0]
    
    # Strategy:
    # 1. Determine reasonable range for m (size of set).
    #    Larger m increases |U+U| and |U-U| but also increases max U (denominator).
    #    The term log(ratio)/log(max U) suggests that if ratio grows polynomially with m
    #    and max U grows exponentially with m (for b>1), the score might decrease for large m.
    #    However, for b close to 1, max U grows slowly.
    #    Let's try m from 2 to 20.
    
    # 2. For each m, search over b.
    #    b must be > 1.
    #    If b is very close to 1, elements are close, many collisions in sums/diffs?
    #    If b is large, elements are sparse.
    #    Let's sample b in [1.1, 10.0] initially.
    
    # We will use a nested loop:
    # Outer: m
    # Inner: b (grid search + local refinement)
    
    # To be efficient, we can pre-compute or just rely on the speed of _score_of for small m.
    # m=20, |U|=20, |U+U| <= 400. Set operations are very fast.
    # We can afford many evaluations.
    
    # Let's do a coarse grid first.
    m_range = range(2, 15)
    b_coarse = [1.0 + i * 0.1 for i in range(1, 100)] # 1.1 to 10.0
    
    # Also include integer bases specifically as they are often optimal for such structures
    b_int = [2, 3, 4, 5, 10]
    b_coarse.extend(b_int)
    
    # Deduplicate b_coarse
    b_coarse = sorted(set(b_coarse))
    
    # Phase 1: Coarse Grid Search
    for m in m_range:
        if time.time() > deadline:
            break
        for b in b_coarse:
            if time.time() > deadline:
                break
            U = _build_geometric(b, m)
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U
                best_m = m
                best_b = b
    
    # Phase 2: Local Refinement around the best found (best_m, best_b)
    if best_m is not None and best_b is not None:
        # Refine m around best_m
        m_min = max(2, best_m - 2)
        m_max = best_m + 3
        
        # Refine b around best_b
        # If best_b was from coarse grid, refine in a small window
        # If best_b was integer, maybe check nearby floats?
        # Let's just do a fine grid around the current best.
        
        # Determine step size for refinement
        # If best_b is large, use larger steps? No, relative step.
        # Let's use a fixed number of steps in a relative range.
        # Range: [best_b * 0.9, best_b * 1.1]
        
        b_low = best_b * 0.9
        b_high = best_b * 1.1
        # Ensure b_low > 1
        if b_low <= 1.0:
            b_low = 1.01
            
        # Also check integer neighbors if best_b is close to an integer
        b_candidates_fine = []
        # Fine grid
        num_steps = 100
        step = (b_high - b_low) / num_steps
        for i in range(num_steps + 1):
            b_candidates_fine.append(b_low + i * step)
            
        # Add integer neighbors if applicable
        b_int_neighbor = round(best_b)
        for nb in [b_int_neighbor - 1, b_int_neighbor, b_int_neighbor + 1]:
            if nb > 1:
                b_candidates_fine.append(float(nb))
                
        b_candidates_fine = sorted(set(b_candidates_fine))
        
        for m in range(m_min, m_max + 1):
            if time.time() > deadline:
                break
            for b in b_candidates_fine:
                if time.time() > deadline:
                    break
                U = _build_geometric(b, m)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                    best_m = m
                    best_b = b

    # Phase 3: Even finer refinement if time permits
    if time.time() < deadline - 5.0 and best_m is not None and best_b is not None:
        b_low = best_b * 0.95
        b_high = best_b * 1.05
        if b_low <= 1.0:
            b_low = 1.001
            
        num_steps = 200
        step = (b_high - b_low) / num_steps
        for m in [best_m - 1, best_m, best_m + 1]:
            if m < 2: continue
            if time.time() > deadline:
                break
            for i in range(num_steps + 1):
                if time.time() > deadline:
                    break
                b = b_low + i * step
                U = _build_geometric(b, m)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                    best_m = m
                    best_b = b

    return best_set
# EVOLVE-BLOCK-END
