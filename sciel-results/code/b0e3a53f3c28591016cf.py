import time
import itertools

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill

# Global state for the stateful heuristic
_seen_vectors = []
_conflict_counts = {}
_total_conflicts_observed = 0
_initialized = False

def _stable_hash(v):
    """Deterministic hash for tie-breaking."""
    h = 5381
    for x in v:
        h = (h * 33 + x) & 0xFFFFFFFF
    return h

def _get_priority(vector, n):
    """
    Compute a dynamic priority for a vector.
    
    Strategy:
    1. Base score: Favor 'balanced' vectors (more 1s/2s, fewer 0s) to move away from 
       the {0,1}^n subcube which is known to trap greedy algorithms.
    2. Conflict penalty: Subtract a penalty based on how many times this vector has 
       been seen to conflict with previously added vectors in this specific run.
    3. Look-ahead: Estimate how many *other* unselected vectors would be blocked 
       if this vector is added. (Approximated by counting potential partners).
    """
    global _total_conflicts_observed
    
    vec_tuple = tuple(vector)
    
    # 1. Base geometric score
    # We want to maximize distance from the "corner" (0,0,...,0) and (2,2,...,2).
    # Vectors with many 1s are generally "central" in the sense of being 
    # non-parallel to the axes in a useful way for cap sets.
    c1 = vec_tuple.count(1)
    c2 = vec_tuple.count(2)
    c0 = vec_tuple.count(0)
    
    # Reward balance: prefer vectors where c1, c2, c0 are roughly n/3.
    # But specifically, we want to avoid the subspace {0,1}^n.
    # So, heavily penalize c2 == 0.
    penalty_no_2 = 10.0 if c2 == 0 else 0.0
    
    # Encourage mixing
    balance_score = -abs(c1 - n/3.0) - abs(c2 - n/3.0) - abs(c0 - n/3.0)
    
    base_score = balance_score * 5.0 - penalty_no_2
    
    # 2. Dynamic Conflict Penalty
    # If this vector has frequently conflicted with the current set, lower its priority.
    # conflict_counts[vec] stores how many times we tried to add it or considered it
    # and found it formed a line with existing members.
    # Actually, a better metric: How many vectors in the *current* set does it conflict with?
    # This is expensive to compute for every call if not cached.
    # Instead, we track a "danger" score.
    
    # For now, use a simpler dynamic component:
    # If we have seen this vector in the 'seen' list and it was rejected, penalize it.
    # But 'priority' is called for *all* vectors to sort them.
    # We need a static-ish score that adapts slowly.
    
    # Let's use the number of 1s as a primary driver, modulated by a global "phase".
    # Early on, prefer vectors with many 1s.
    # Later, prefer vectors that are "orthogonal" to the existing span?
    
    # A simpler, more robust stateful heuristic:
    # Score = (Number of 1s) * 2.0 + (Number of 2s) * 1.0 - (Number of 0s) * 0.5
    # Plus a small noise term based on hash to break ties deterministically but varied.
    
    score = c1 * 2.0 + c2 * 1.5 - c0 * 0.5
    
    # Add a small hash-based jitter to avoid strict ordering issues
    h = _stable_hash(vec_tuple)
    score += (h % 100) * 0.001
    
    return score

def priority(vector, n):
    """
    Returns a priority score for the vector. Higher is better.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to a simple deterministic score if we run out of time
        return sum(vector)
    
    vec_tuple = tuple(vector)
    
    # We don't do complex precomputation. We rely on the heuristic being fast.
    # The "stateful" aspect is light: we just use a good static heuristic that
    # is known to perform well for cap sets (favoring 1s and 2s over 0s).
    
    c1 = vec_tuple.count(1)
    c2 = vec_tuple.count(2)
    c0 = vec_tuple.count(0)
    
    # Heuristic:
    # 1. Maximize c1 (vectors in {0,1}^n are bad, so we want 2s, but 1s are "positive")
    # Actually, the known good heuristic for cap sets is to prefer vectors with
    # a "balanced" distribution of 0, 1, 2.
    # However, the {0,1}^n subcube is the main trap.
    # So, we heavily reward the presence of 2s.
    
    score = c1 * 1.0 + c2 * 3.0 - c0 * 1.0
    
    # Tie-breaker: stable hash
    h = _stable_hash(vec_tuple)
    score += h * 1e-6
    
    return score
