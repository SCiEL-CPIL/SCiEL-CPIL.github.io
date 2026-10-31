"""Cap set greedy priority function with pair-sum line-completion tracking."""
import hashlib
import time

# EVOLVE-BLOCK-START

# Global state for tracking pair sums and deadlines
_state = {
    'initialized': False,
    'n': 0,
    'pair_sums': {},  # maps tuple -> count of pairs in current set that sum to this
    'set_vectors': set(),  # vectors currently in the cap set
    'deadline': 0.0,
    'start_time': 0.0,
    'total_vectors': 0
}

def _reset_state(n, start_time):
    """Reset global state for a new evaluation."""
    _state['initialized'] = True
    _state['n'] = n
    _state['pair_sums'] = {}
    _state['set_vectors'] = set()
    _state['deadline'] = start_time + 290.0  # 10s safety margin before 300s limit
    _state['start_time'] = start_time
    _state['total_vectors'] = 0

def _add_to_set(vector, n):
    """Add a vector to the cap set, updating pair sums."""
    vec_tuple = tuple(vector)
    _state['set_vectors'].add(vec_tuple)
    _state['total_vectors'] += 1
    
    # For each existing vector in the set, compute the sum and update pair_sums
    for existing in _state['set_vectors']:
        if existing == vec_tuple:
            continue
        # Sum in F_3: (a + b) % 3
        pair_sum = tuple((existing[i] + vec_tuple[i]) % 3 for i in range(n))
        _state['pair_sums'][pair_sum] = _state['pair_sums'].get(pair_sum, 0) + 1

def _remove_from_set(vector, n):
    """Remove a vector from the cap set, updating pair sums."""
    vec_tuple = tuple(vector)
    if vec_tuple not in _state['set_vectors']:
        return
    _state['set_vectors'].discard(vec_tuple)
    _state['total_vectors'] -= 1
    
    # For each remaining vector in the set, decrement the pair sum count
    for existing in list(_state['set_vectors']):
        pair_sum = tuple((existing[i] + vec_tuple[i]) % 3 for i in range(n))
        if pair_sum in _state['pair_sums']:
            _state['pair_sums'][pair_sum] -= 1
            if _state['pair_sums'][pair_sum] <= 0:
                del _state['pair_sums'][pair_sum]

def _fast_hash(vector):
    """Fast hash for ordering vectors deterministically."""
    # Use SHA256 for better uniformity
    h = hashlib.sha256(bytes(vector)).digest()
    # Take first 8 bytes as a 64-bit integer
    return int.from_bytes(h[:8], 'big')

def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Uses pair-sum tracking to detect line-completion risk:
    - If adding this vector would complete a 3-term AP, penalize heavily
    - Prefer vectors with lower line-completion risk
    - Use balanced coordinates as tiebreaker
    """
    # Initialize state on first call
    if not _state['initialized'] or _state['n'] != n:
        _reset_state(n, time.time())
    
    # Check deadline - if we're running out of time, return a neutral priority
    if time.time() > _state['deadline']:
        return 0.0
    
    vec_tuple = tuple(vector)
    
    # Count how many existing vectors would form a 3-AP with this vector
    # A 3-AP exists if there's an existing vector u such that (u + v) % 3 is also in the set
    # Equivalently: for each existing u, check if (2*v - u) % 3 is in the set
    # Or: count pair sums that would be completed
    
    line_completion_risk = 0
    for existing in _state['set_vectors']:
        # Check if adding vector completes a line with existing
        # In F_3, three points a, b, c form an AP iff a + c = 2*b (mod 3)
        # So if we have existing u and new v, they form an AP with some w if:
        # w = (2*v - u) % 3 or w = (2*u - v) % 3
        # Check if such w is already in the set
        for i in range(n):
            pass  # Placeholder - we'll use pair sums instead
        
        # Using pair sums: if (existing + v) % 3 has count > 0, that means
        # there's another vector w such that existing + w = (existing + v) % 3
        # which means w = v, so no new completion from this pair alone
        # Actually, we need to check if there exists w in set such that
        # w = (2*v - existing) % 3 or w = (2*existing - v) % 3
        
        # Simpler: check if the "third point" of the AP is in the set
        # For AP: a, b, c where b = (a+c)/2 = (a+c)*2 mod 3 (since 2*2=1 mod 3)
        # So if we have a=existing, b=v, then c = (2*v - existing) % 3
        # Or a=v, b=existing, then c = (2*existing - v) % 3
        third1 = tuple((2 * vec_tuple[i] - existing[i]) % 3 for i in range(n))
        third2 = tuple((2 * existing[i] - vec_tuple[i]) % 3 for i in range(n))
        
        if third1 in _state['set_vectors'] or third2 in _state['set_vectors']:
            line_completion_risk += 1
    
    # Coordinate balance as secondary criterion
    count_0 = count_1 = count_2 = 0
    for val in vec_tuple:
        if val == 0:
            count_0 += 1
        elif val == 1:
            count_1 += 1
        elif val == 2:
            count_2 += 1
    
    min_count = min(count_0, count_1, count_2)
    imbalance = abs(count_0 - count_1) + abs(count_1 - count_2) + abs(count_0 - count_2)
    balance_score = float(min_count) - 0.01 * float(imbalance)
    
    # Heavy penalty for line completion risk
    # Use a large negative weight to strongly discourage completing lines
    risk_penalty = -10.0 * float(line_completion_risk)
    
    # Add small deterministic tiebreaker based on hash
    tiebreaker = _fast_hash(vec_tuple) % 1000 * 0.001
    
    return balance_score + risk_penalty + tiebreaker

# EVOLVE-BLOCK-END
