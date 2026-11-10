import numpy as np
import time
import random
import sys

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 10  # vertical direction index

def _encode(xa, xb, ya, yb):
    return ((xa * 7 + xb) * 49) + (ya * 7 + yb)

def _xy_from_slot(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

# Precompute all points
_PTS = np.zeros((_SLOTS, 4), dtype=np.uint8)
for s in range(_SLOTS):
    xa, xb, ya, yb = _xy_from_slot(s)
    _PTS[s] = [xa, xb, ya, yb]

# Precompute lines for each direction
# lines[d] is a list of 49 arrays, each of length 49 (the points on that line)
_lines = []
# line_of[d][s] = index of the line in _lines[d] that contains point s
_line_of = []

for d in range(50):
    if d == _INF:
        # v = (0,1): lines are vertical, grouped by x
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for xa in range(7):
            for xb in range(7):
                line = []
                for ya in range(7):
                    for yb in range(7):
                        s = _encode(xa, xb, ya, yb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)
    else:
        da, db = (d // 7, d % 7)
        # v = (1, da + db*alpha)
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for ya0 in range(7):
            for yb0 in range(7):
                line = []
                for ta in range(7):
                    for tb in range(7):
                        # t = ta + tb*alpha
                        # x = 0 + t = t
                        # y = y0 + t*(da + db*alpha)
                        txa = (ta * da + tb * db * _kq) % _p
                        txb = (ta * db + tb * da) % _p
                        nxa = ta % _p
                        nxb = tb % _p
                        nya = (ya0 + txa) % _p
                        nyb = (yb0 + txb) % _p
                        s = _encode(nxa, nxb, nya, nyb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)

def _compute_Scount(S):
    """Compute Scount[d][s] = number of S points on line through s in direction d (including s if in S)."""
    Scount = np.zeros((50, _SLOTS), dtype=np.int16)
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    for d in range(50):
        for line in _lines[d]:
            c = np.sum(S_arr[line])
            Scount[d, line] = c
    return Scount

def _is_valid(S):
    """
    Check if S (complement of N) gives a valid Nikodym set.
    Condition: For every point x, there exists a direction d such that the line through x in direction d
    has no S-points other than x.
    i.e., Scount[d][x] - S[x] == 0.
    """
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    Scount = _compute_Scount(S)
    for s in range(_SLOTS):
        ss = S_arr[s]
        ok = False
        for d in range(50):
            if Scount[d, s] - ss == 0:
                ok = True
                break
        if not ok:
            return False
    return True

def _to_array(S):
    """Convert S (bytearray or bytes) to the output array of N = complement of S."""
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    pts = []
    for s in range(_SLOTS):
        if not S_arr[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def _random_greedy_MIS(deadline):
    """
    Randomized Greedy Maximum Independent Set on the conflict graph.
    We want to maximize |S| such that S is a valid excluded set.
    
    Validity constraint:
    For all x, exists d s.t. (Line_d(x) \ {x}) ∩ S = ∅.
    
    This is equivalent to saying that for every x, at least one direction d is "clean" for x.
    A direction d is clean for x if no other point on Line_d(x) is in S.
    
    Let's define a conflict between two points u and v if they share a line in direction d,
    and if both u and v are in S, it might cause issues?
    Actually, the constraint is global per point x.
    
    Alternative view:
    Let C_x be the set of constraints for point x.
    Constraint for x is satisfied if ∃ d ∈ {0..49} such that S ∩ (Line_d(x) \ {x}) = ∅.
    
    This is a CSP. We can solve it via greedy assignment.
    We assign points to S or N one by one.
    When we assign a point p to S, it "blocks" all other points q on the same line (in any direction)
    from using that specific line-direction pair as their "clean" direction.
    
    Let's maintain for each point x a set of available clean directions A_x.
    Initially A_x = {0, ..., 49} for all x.
    When we add p to S:
    For each direction d, let L = Line_d(p).
    For each q ∈ L \ {p}, d is no longer a valid clean direction for q.
    So we remove d from A_q.
    If A_q becomes empty, q is "invalidated" (cannot be satisfied).
    
    We want to maximize |S|.
    Greedy heuristic:
    1. Start with S = ∅, A_x = all directions for all x.
    2. While time remains:
       a. Find a point p not in S and not in N (unassigned) such that adding it to S
          causes the minimum number of "critical" invalidations.
          Critical invalidation: q becomes invalid (A_q becomes empty).
       b. If no such point exists (all remaining points would cause invalidation), stop?
          Or maybe we should just pick the one that causes the least "damage" (reduction in total available directions).
       c. Add p to S.
       d. Update A_q for all q on lines through p.
       
    However, we also need to ensure that points in N (not in S) are valid.
    Wait, the validity condition applies to ALL points, whether in S or N.
    If x ∈ N, it needs a clean direction.
    If x ∈ S, it needs a clean direction (the line has no *other* S points).
    
    So the algorithm above works for both.
    We just need to ensure that at the end, all points have A_x non-empty.
    If we stop when no point can be added to S without making someone's A_x empty,
    we have a maximal valid S?
    Not necessarily maximal, but a valid one.
    
    To maximize |S|, we should prioritize adding points that are "safe".
    
    Let's refine:
    We maintain `avail_dirs[x]` as a bitmask of 50 bits (using Python ints or uint64).
    Initially all 1s.
    
    When adding p to S:
    For each d in 0..49:
        line = _lines[d][_line_of[d][p]]
        For each q in line:
            if q != p:
                avail_dirs[q] &= ~(1 << d)
                if avail_dirs[q] == 0:
                    # q is now unsatisfiable
                    # This is bad. We should avoid this.
                    pass
    
    So, before adding p, we check:
    Would any q have avail_dirs[q] become 0?
    For each d, for each q on line_d(p) \ {p}:
        if (avail_dirs[q] & ~(1 << d)) == 0 and (avail_dirs[q] & (1 << d)) != 0:
            # q's only remaining direction is d, and we are removing it
            # So q becomes invalid.
            # Count this as a "violation".
    
    We pick p with minimum violations (ideally 0).
    If multiple, pick one with minimum total reduction in available directions for neighbors.
    
    This is a greedy set packing / independent set heuristic.
    """
    S = bytearray(_SLOTS)
    avail_dirs = [0xFFFFFFFFFF] * _SLOTS  # 50 bits set
    
    # Precompute list of points for fast iteration
    all_points = list(range(_SLOTS))
    
    # To speed up, we can track "degree" or "impact"
    # But simple O(N * 50 * 49) per step is 2401 * 2450 ~ 6M ops per step.
    # 2401 steps * 6M = 14B ops. Too slow in Python.
    
    # We need a faster way.
    # Instead of iterating all points each time, we can use a priority queue?
    # Or just random sampling?
    
    # Let's try a randomized restart approach with a faster local check.
    # Or, since the structure is regular, maybe we can just randomize the order of points
    # and greedily add them if they don't break validity.
    
    # Random Order Greedy:
    # 1. Shuffle all points.
    # 2. For each point p in shuffled order:
    #    Check if adding p to S keeps all points valid.
    #    To check validity efficiently:
    #    We maintain avail_dirs.
    #    If we tentatively add p, we update avail_dirs for neighbors.
    #    If any neighbor becomes 0, revert and skip p.
    #    Otherwise, keep p in S.
    
    # This is O(N * (cost of check)).
    # Cost of check: For each d, for each q in line_d(p)\{p}, check if (avail_dirs[q] & ~(1<<d)) == 0.
    # This is 50 * 48 = 2400 checks.
    # 2401 * 2400 = ~6M checks total. This is feasible in Python if optimized.
    
    best_S = None
    best_size = 0
    
    rng = random.Random(42)
    
    for restart in range(10): # Do a few restarts
        if time.time() > deadline - 5: break
        
        S_try = bytearray(_SLOTS)
        avail_dirs = [0xFFFFFFFFFF] * _SLOTS
        points = list(range(_SLOTS))
        rng.shuffle(points)
        
        size = 0
        for p in points:
            # Check if adding p is safe
            safe = True
            # We need to check if any q loses its last direction
            # We also need to store the changes to revert if unsafe
            
            changes = []
            for d in range(50):
                line_idx = _line_of[d][p]
                line = _lines[d][line_idx]
                mask = 1 << d
                for q in line:
                    if q == p:
                        continue
                    # If q currently has direction d available, and no other directions
                    # then removing d makes q invalid.
                    if (avail_dirs[q] & mask) != 0:
                        # d is currently available for q
                        # Check if d is the ONLY available direction
                        if (avail_dirs[q] & ~mask) == 0:
                            safe = False
                            break
                        # Else, q will still have other directions, so it's fine to remove d
                        # We record the change
                        changes.append((q, d))
                if not safe:
                    break
            
            if safe:
                # Apply changes
                for q, d in changes:
                    avail_dirs[q] &= ~(1 << d)
                S_try[p] = 1
                size += 1
            # If not safe, we skip p (it goes to N)
        
        if size > best_size:
            best_size = size
            best_S = S_try[:]
            
        if time.time() > deadline - 5: break

    if best_S is None:
        # Fallback: empty S is always valid?
        # If S is empty, N is all points.
        # For any x, line through x has 0 other points in S. So valid.
        best_S = bytearray(_SLOTS)
        best_size = 0
        
    return best_S, best_size

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 580  # safety margin
    
    # Try the randomized greedy MIS
    S, size = _random_greedy_MIS(deadline)
    
    # Verify validity just in case
    if not _is_valid(bytes(S)):
        # Fallback to empty S (valid)
        S = bytearray(_SLOTS)
    
    return _to_array(bytes(S))
