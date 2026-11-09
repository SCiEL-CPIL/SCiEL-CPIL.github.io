import time

_P = 1009
_M = 5

# Precompute the best start and roots
def _precompute():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    roots = {}
    for x in range(1, _P):
        if time.time() > deadline:
            break
        r = (x * x) % _P
        if r != 0:
            if r not in roots:
                roots[r] = x
    
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            break
        valid = True
        for i in range(_M):
            if (s + i) not in roots:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
    return None

_RESULT = _precompute()

def construct():
    return _RESULT
