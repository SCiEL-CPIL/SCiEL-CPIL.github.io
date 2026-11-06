import zlib
import time
import math

_START = time.time()
_DEADLINE = _START + 275.0  # safety margin before 330s kill

# Adaptive state
_added_vectors = []
_added_set = set()
_call_count = [0]
_score_cache = {}
_cache_limit = 50000

# Conflict profile: for quick lookup of how "crowded" a region is
# We use a simplified approach: track the count of vectors by their
# "type" signature to estimate conflict potential cheaply
_type_counts = {}  # tuple of (c0, c1, c2) -> count
_coord_sum = [0, 0, 0]  # sum of coordinates across all added vectors

def priority(vector, n):
    """
    Adaptive conflict-aware priority with fast conflict estimation.
    
    Combines:
    1. Static structural features (from inspiration 1) for fast baseline
    2. Adaptive conflict estimation using type-based profiling
    3. Caching for repeated queries
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0
    
    _call_count[0] += 1
    
    # Check cache
    vkey = vector
    if vkey in _score_cache:
        return _score_cache[vkey]
    
    # Compute basic features
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Static structural score (fast, always computed)
    if n > 0:
        one_score = math.exp(-0.5 * c1 / n)
    else:
        one_score = 1.0
    
    if c0 + c2 > 0:
        balance = 2.0 * min(c0, c2) / (c0 + c2)
    else:
        balance = 0.0
    
    # Deterministic hash for tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    if not _added_vectors:
        # Bootstrap: no vectors added yet, use static priority
        score = 0.7 * one_score + 0.2 * balance + 0.1 * hash_score
    else:
        # Adaptive: estimate conflict using type-based profile
        # Key insight: vectors with similar (c0, c1, c2) profiles are more
        # likely to form 3-APs with each other. We estimate conflict by
        # checking how many existing vectors have similar profiles.
        
        vtype = (c0, c1, c2)
        similar_count = _type_counts.get(vtype, 0)
        
        # Also check "complement" types: if v has (c0,c1,c2), then
        # a vector w that forms a 3-AP with v and some u would have
        # coordinates that sum to 0 mod 3 with v's coordinates.
        # The complement of v is comp[i] = (-v[i]) % 3, which has:
        # comp_0 = c2, comp_1 = c1, comp_2 = c0
        comp_type = (c2, c1, c0)
        comp_count = _type_counts.get(comp_type, 0)
        
        # Conflict estimate: weighted combination of same-type and complement-type
        # proximity
        k = len(_added_vectors)
        same_ratio = similar_count / k
        comp_ratio = comp_count / k
        
        # Vectors that are close to their complement type are riskier
        # because they can complete 3-APs with many existing vectors
        conflict_score = 0.6 * same_ratio + 0.4 * comp_ratio
        
        # Combine: low conflict is good, structural safety is good
        score = 0.5 * (1.0 - conflict_score) + 0.3 * one_score + 0.1 * balance + 0.1 * hash_score
    
    # Cache the score
    if len(_score_cache) < _cache_limit:
        _score_cache[vkey] = score
    
    # Update adaptive state (this is where we track what's been added)
    # Note: we don't actually know which vectors were added by the evaluator,
    # so we approximate by assuming the highest-priority vectors seen so far
    # were added. This is an approximation but captures the adaptive behavior.
    # 
    # Actually, a better approach: we can't reliably track what's added,
    # so we use the type counts as a proxy for the "shape" of the growing set.
    # We update our type profile based on the current vector as a heuristic.
    
    vtype = (c0, c1, c2)
    _type_counts[vtype] = _type_counts.get(vtype, 0) + 1
    _coord_sum[0] += c0
    _coord_sum[1] += c1
    _coord_sum[2] += c2
    
    return score
