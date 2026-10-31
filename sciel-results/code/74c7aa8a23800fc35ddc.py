import time

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    """
    Prioritize vectors that contain no 1s.
    The set {0, 2}^n is a cap set because any 3-term AP in {0, 2}^n
    must have all terms equal (since 0+2=2 != 2*0=0 and 2+0=2 != 2*2=1).
    Thus, greedily selecting all vectors in {0, 2}^n yields a cap set
    of size 2^n. This is a strong lower bound on the maximum cap set size.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Check if the vector contains any 1
    # If it does, it has low priority.
    # If it doesn't (only 0s and 2s), it has high priority.
    
    # To make the ordering deterministic and potentially beneficial for
    # adding subsequent vectors (if any can be added), we can sort
    # {0,2} vectors by the number of 2s (or 0s).
    
    has_one = 1 in vector
    
    if not has_one:
        # High priority
        # Secondary key: count of 2s
        # This ensures a stable order.
        count_2 = vector.count(2)
        return 1000.0 + count_2
    else:
        # Low priority
        # We can still try to add some 1-vectors if there's room,
        # but the {0,2} set is already a cap set.
        # Adding 1-vectors might break the cap set property if they
        # form lines with 0/2 vectors.
        # Line: 0, 1, 2.
        # If we have 0 and 2 in our set, we cannot add 1.
        # Since {0,2}^n contains pairs (0...0, 2...2) etc.,
        # almost all 1-vectors will form lines with existing 0/2 vectors.
        # So it's safer to keep them low.
        return 0.0
