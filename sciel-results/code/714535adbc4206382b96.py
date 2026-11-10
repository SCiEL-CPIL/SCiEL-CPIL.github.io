import numpy as np
import time

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 49

_PTS_XA = np.empty(_SLOTS, dtype=np.uint8)
_PTS_XB = np.empty(_SLOTS, dtype=np.uint8)
_PTS_YA = np.empty(_SLOTS, dtype=np.uint8)
_PTS_YB = np.empty(_SLOTS, dtype=np.uint8)

def _encode(xa, xb, ya, yb):
    return ((xa*7+xb)*49)+(ya*7+yb)

def _xy_from_slot(s):
    yb = (s % 7); s //= 7
    ya = (s % 7); s //= 7
    xb = (s % 7); s //= 7
    xa = (s % 7)
    return xa, xb, ya, yb

# Fill point coordinate arrays
for xa in range(7):
    for xb in range(7):
        base_x = (xa*7+xb)
        for ya in range(7):
            for yb in range(7):
                s = base_x*49+(ya*7+yb)
                _PTS_XA[s]=xa; _PTS_XB[s]=xb; _PTS_YA[s]=ya; _PTS_YB[s]=yb

# Precompute the 50 directions and their 48 step-points for every slot.
_dir_steps = []
_dir_index_at = []
for d in range(50):
    steps = []
    index_at = []
    for s in range(2401):
        xa, xb, ya, yb = _xy_from_slot(s)
        arr = np.empty(48, dtype=np.int32)
        for t in range(1, 49):
            if d == _INF:
                nxa = xa; nxb = xb
                nya = (ya+t) % 7; nyb = (yb+t) % 7
            else:
                da, db = (d//7, d%7)
                nxa = (xa+t) % 7
                if db == 0:
                    nxb = xb
                else:
                    inv = pow(db, 6, 7)
                    nxb = (xb + t*da*inv) % 7
                nya = (ya+t*da) % 7
                nyb = (yb+t*db) % 7
            arr[t-1] = _encode(nxa, nxb, nya, nyb)
        steps.append(arr)
        idx = {}
        for t in range(1, 49):
            idx[int(arr[t-1])] = t
        index_at.append(idx)
    _dir_steps.append(steps)
    _dir_index_at.append(index_at)

def _construct_greedy(seed):
    rnd = np.random.RandomState(seed)
    S = np.zeros(_SLOTS, dtype=bool)
    
    # For each direction d, we have 49 lines.
    # We represent a line by its canonical anchor (min slot) and its 49 points.
    # Actually, to make it fast, we can just iterate over all 50*49 = 2450 lines.
    # Let's precompute the lines.
    # lines[d] = list of 49 arrays of 49 points (int32)
    lines = []
    for d in range(50):
        seen = set()
        d_lines = []
        for s in range(2401):
            if s in seen:
                continue
            # find canonical rep
            # line through s in dir d
            # points are s and _dir_steps[d][s]
            # we want the min slot in this set
            # actually we can just add s and all its steps, find min, and store
            # but we need to be careful not to add duplicates
            # let's just build the set
            line_pts = np.concatenate(([s], _dir_steps[d][s]))
            rep = np.min(line_pts)
            if rep in seen:
                continue
            seen.update(line_pts)
            d_lines.append(line_pts)
        lines.append(d_lines)
    
    # uncovered: boolean array, True if not in S
    uncovered = np.ones(_SLOTS, dtype=bool)
    
    # We want to maximize S.
    # Greedy: pick a line that covers the most uncovered points, add all its points to S.
    # Repeat until no line covers > 0 uncovered points? No, we can add lines even if they cover 0, but it's useless.
    # Actually, we MUST ensure that the final S is valid.
    # Validity: every point x (in N or S) is on a line with >= 2 other S points.
    # If we just greedily add lines to S, we might end up with a valid S.
    # Let's just add lines until all points are covered by S? No, S can be all points except N.
    # If S is all points, N is empty, invalid.
    # So we should stop when S is large enough, or just add lines until we can't add more that cover new points?
    # If we add lines until every point is in S, then S is all points.
    # We need to leave some points in N.
    # Wait, the metric is 2401 - |N| = |S|. So we want S as large as possible.
    # But S cannot be all 2401 points.
    # Can S be 2400 points? If N has 1 point x, then x must be on a line with >= 2 other S points.
    # If N={x}, then for x to be valid, there must be a line through x with >= 2 other points in S.
    # Since all other points are in S, any line through x has 48 other points in S. So x is valid.
    # What about other points y in S? They must also be on a line with >= 2 other S points.
    # Since S is huge, this is easily satisfied.
    # So S = all points except one is valid!
    # Wait, is it?
    # Let's check: N = {x}. S = F_q^2 \ {x}.
    # For any point z in F_q^2:
    # If z = x, it is on a line with 48 other points in S. Valid.
    # If z != x, it is in S. It is on a line. Does that line have >= 2 other points in S?
    # The line has 49 points total. One is z.
    # If the line does not contain x, it has 48 other points, all in S. Valid.
    # If the line contains x, it has 47 other points (excluding z and x), all in S. Valid.
    # So YES, N = {x} is a valid Nikodym set!
    # And |N| = 1, so |S| = 2400.
    # The score is 2400.
    # The current program gets 1.0000, which corresponds to a much smaller S.
    # The inspiration says "Greedy set cover minimizes R, maximizing S size to ~300 vs 147 Measured: 1 -> 1.537."
    # Wait, if N={x} is valid, why did the inspiration only get ~300?
    # Maybe I misread the validity condition.
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    # Ah! "whose points other than x all belong to N".
    # NOT S! The line's other points must be in N!
    # So N is the Nikodym set. The points on the line (other than x) must be in N.
    # So we want N to be small, but it must "cover" the plane in the sense that every point is on a line whose other points are in N.
    # This is the standard definition of a Nikodym set: a set N such that for every point x, there is a line L through x such that L \ {x} subset N.
    # So N must be large enough to contain L \ {x} for some line L through x.
    # If N={x}, then for any other point y, we need a line through y such that L \ {y} subset {x}.
    # But L \ {y} has 48 points, and {x} has 1 point. Impossible.
    # So N cannot be {x}. N must be much larger.
    # The inspiration says S (which is the complement of N) is ~300. So |N| is ~2100.
    # Wait, the metric is 2401 - |N| = |S|. So we want S to be large.
    # If S is ~300, then |N| is ~2100.
    # The current program gets 1.0000, which means S is small (maybe ~147 as per inspiration).
    # So we want to maximize S.
    # Let's re-read the inspiration: "Greedy set cover minimizes R, maximizing S size to ~300 vs 147".
    # So S is the set of points NOT in N.
    # We want to maximize S.
    # Let's implement a greedy algorithm to maximize S.
    # We start with S empty. We want to add points to S.
    # A point x can be in S if it is "covered" by N.
    # Wait, the condition is: for every point x (in N or S), there is a line L through x such that L \ {x} subset N.
    # This must hold for ALL points.
    # If we want to maximize S, we want to put as many points as possible into S.
    # A point x can be in S only if there exists a line L through x such that L \ {x} subset N.
    # But N is the complement of S. So L \ {x} must be disjoint from S.
    # So x can be in S if there is a line through x such that all other 48 points on that line are NOT in S.
    # This is a self-referential condition.
    # Let's think about it differently.
    # We want to find a set N such that for every x, there is a line L through x with L \ {x} subset N.
    # We want to minimize |N|.
    # This is a covering problem.
    # Each line L "covers" the points x on L, in the sense that if L \ {x} subset N, then x can be in S.
    # Actually, if we choose a set of lines, and for each line L, we require that L \ {x} subset N for all x on L? No, just for one x.
    # Wait, the condition is: for EVERY x, there EXISTS a line L_x through x such that L_x \ {x} subset N.
    # So N must be such that for every x, there is a line through x that is "almost contained" in N (missing only x).
    # If we want to minimize N, we can try to cover the plane with lines such that each line L allows its points to be in S, provided the other points are in N.
    # This is tricky.
    # Let's look at the current program. It uses a baseline construction and then tries to improve it.
    # The baseline is based on a unital.
    # Let's just implement a simple greedy that tries to maximize S by adding points that are "safe".
    # A point x is safe to add to S if there is a line through x such that all other points on that line are NOT in S.
    # We can start with S empty. Then every point is safe (since all lines have 0 other points in S).
    # So we can add all points to S? But then N is empty, which is invalid.
    # Wait, if S is all points, then for any x, we need a line L through x such that L \ {x} subset N = empty.
    # But L \ {x} has 48 points, so it's not empty. So x is NOT covered.
    # So S cannot be all points.
    # In fact, if x is in S, then for x to be valid, there must be a line L through x such that L \ {x} subset N.
    # This means that if x is in S, then at least one line through x must have its other 48 points in N.
    # So if we put x in S, we are "forcing" 48 points to be in N (at least for one line).
    # To minimize N, we should choose lines that share many points.
    # This is a set cover problem where we want to cover all points x with lines L_x, and the cost is the union of (L_x \ {x}) for all x.
    # Actually, N must contain L_x \ {x} for the chosen L_x for each x.
    # So N is the union of (L_x \ {x}) for all x.
    # We want to minimize |N|.
    # This is equivalent to maximizing |S|, where S is the set of points x such that we have chosen a line L_x with L_x \ {x} subset N.
    # Wait, if x is in S, then L_x \ {x} must be in N.
    # If x is in N, it doesn't matter.
    # So we can think of it as: we want to choose for each x a line L_x through x, such that the set N = union (L_x \ {x}) is as small as possible.
    # Then S = complement of N.
    # And we must ensure that for x in S, L_x \ {x} is indeed in N.
    # But if N is defined as the union, then it is.
    # So the problem reduces to: choose for each x a line L_x through x, to minimize |union (L_x \ {x})|.
    # This is a very different framing!
    # Let's test this. If we choose L_x to be the same line for all x on that line, then for a line L, N contains L \ {x} for each x in L.
    # So N contains L \ {x1} U L \ {x2} ... = L \ {x1} U (L \ {x2}) ...
    # If L has 49 points, L \ {x} has 48 points.
    # The union of L \ {x} for all x in L is L (since for any y in L, y is in L \ {x} for any x != y).
    # So if we choose L_x = L for all x in L, then N contains L.
    # So if we partition the plane into lines, and for each line L, we choose L_x = L for all x in L, then N is the union of all these lines, which is the whole plane.
    # So N = F_q^2, S = empty. |S| = 0.
    # We want to minimize N, so we want to choose L_x such that the union of (L_x \ {x}) is small.
    # If we can make N small, then S is large.
    # This is a known problem: finding a small Nikodym set.
    # The current program gets S ~ 147. The inspiration gets S ~ 300.
    # Let's implement a greedy that tries to minimize N.
    # We can start with N empty. We want to add points to N.
    # A point x is "satisfied" if there is a line L through x such that L \ {x} subset N.
    # We want all points to be satisfied.
    # We can greedily add points to N that satisfy the most unsatisfied points.
    # This is a standard greedy set cover.
    # Let's implement this.

def _construct_greedy_n(seed):
    rnd = np.random.RandomState(seed)
    N = np.zeros(_SLOTS, dtype=bool)
    satisfied = np.zeros(_SLOTS, dtype=bool)
    
    # Precompute for each point x, the list of lines through x.
    # A line is identified by (d, anchor).
    # We can just use the _dir_steps and _dir_index_at.
    # For a point x and direction d, the line is the set {x} U _dir_steps[d][x].
    # We need to check if L \ {x} subset N.
    # This is equivalent to: for all v in _dir_steps[d][x], N[v] is True.
    
    # We want to add points to N to satisfy as many points as possible.
    # Initially, no point is satisfied (since N is empty).
    # We can pick a point to add to N. Which point?
    # Adding a point p to N might satisfy some points x for which p is on L_x \ {x}.
    # Actually, adding p to N helps satisfy x if p is in L_x \ {x} for some line L_x through x.
    # This is complex.
    # Let's use the other framing: we want to choose L_x for each x to minimize |union (L_x \ {x})|.
    # This is hard to do directly.
    # Let's stick to the greedy set cover on N.
    # We want to cover all points x with "witnesses" (L_x \ {x}).
    # A witness for x is a set of 48 points (a line minus x).
    # We want to choose a collection of witnesses such that every x is covered, and the union of the witnesses is minimized.
    # Wait, the union of the witnesses IS N.
    # So we want to minimize |N| such that for every x, there is a witness W_x (which is L_x \ {x}) with W_x subset N.
    # This is equivalent to: N must contain at least one witness for each x.
    # So N must be a hitting set for the family of all possible witnesses? No.
    # For each x, N must contain at least one of the 50 witnesses for x.
    # So N must be a set that intersects the family of witnesses for each x in at least one full witness.
    # Actually, N must CONTAIN a witness for each x.
    # So N must be a superset of some witness for each x.
    # We want to minimize |N|.
    # This is a minimum set cover problem where the sets are the witnesses, and we want to cover all x, but with the constraint that the union of the chosen witnesses is N.
    # Wait, if we choose a witness W_x for each x, then N must contain W_x for all x.
    # So N is the union of the chosen witnesses.
    # We want to minimize |union of chosen witnesses|.
    # This is exactly the problem I identified earlier.
    # Let's implement a greedy that picks witnesses to minimize the union.
    # We can start with N empty. We want to add witnesses to N.
    # But we must choose one witness for each x.
    # This is like assigning each x to a line L_x, to minimize the size of the union of (L_x \ {x}).
    # Let's try a simple heuristic: for each x, pick the line L_x that has the most points already in N.
    # This is a local search.
    # We can start with a random assignment, and then iteratively reassign x to the line that minimizes the increase in |N|.
    # Let's implement this.

def _construct_local_search(seed):
    rnd = np.random.RandomState(seed)
    # Assign each x to a random line (direction d).
    # L_x is the line through x in direction d.
    # We store d_x for each x.
    d_x = np.random.randint(0, 50, size=_SLOTS)
    
    # Compute N as the union of (L_x \ {x}) for all x.
    # We can represent N as a boolean array.
    N = np.zeros(_SLOTS, dtype=bool)
    # For each x, mark all points in L_x \ {x} as True in N.
    for x in range(_SLOTS):
        d = d_x[x]
        for v in _dir_steps[d][x]:
            N[v] = True
    
    # Now we have a valid N (by construction, every x has a witness in N).
    # We want to minimize |N|.
    # We can try to reassign x to a different line to reduce |N|.
    # Reassigning x from d_old to d_new:
    # The old witness W_old = L_x \ {x} in direction d_old.
    # The new witness W_new = L_x \ {x} in direction d_new.
    # N will lose W_old \ W_new and gain W_new \ W_old.
    # We can compute the change in |N|.
    # If |N| decreases, we accept the change.
    
    # To do this efficiently, we can maintain a count for each point v in N: how many x have v in their witness.
    # count[v] = number of x such that v in (L_x \ {x}).
    # Then |N| is the number of v with count[v] > 0.
    # When we reassign x from d_old to d_new:
    # For v in W_old: count[v] -= 1. If count[v] becomes 0, |N| decreases by 1.
    # For v in W_new: count[v] += 1. If count[v] was 0, |N| increases by 1.
    # We can compute the delta in |N|.
    
    count = np.zeros(_SLOTS, dtype=np.int32)
    for x in range(_SLOTS):
        d = d_x[x]
        for v in _dir_steps[d][x]:
            count[v] += 1
    
    current_N_size = np.sum(count > 0)
    
    # Iteratively improve.
    improved = True
    while improved:
        improved = False
        # Shuffle order to avoid bias.
        order = np.random.permutation(_SLOTS)
        for x in order:
            d_old = d_x[x]
            W_old = _dir_steps[d_old][x]
            # Try all other directions.
            best_d = d_old
            best_delta = 0
            for d_new in range(50):
                if d_new == d_old:
                    continue
                W_new = _dir_steps[d_new][x]
                # Compute delta.
                delta = 0
                # For v in W_old: count[v] -= 1
                # If count[v] becomes 0, delta -= 1
                # For v in W_new: count[v] += 1
                # If count[v] was 0, delta += 1
                # We can compute this by checking the counts.
                # But we must be careful: W_old and W_new might overlap.
                # Actually, W_old and W_new are both lines through x, so they intersect only at x.
                # But W_old and W_new exclude x, so they are disjoint!
                # Wait, is that true?
                # Two distinct lines through x intersect only at x.
                # So W_old and W_new are disjoint.
                # So we can just sum the effects.
                # For v in W_old: if count[v] == 1, then it will become 0, so delta -= 1.
                # For v in W_new: if count[v] == 0, then it will become 1, so delta += 1.
                # Let's verify this.
                for v in W_old:
                    if count[v] == 1:
                        delta -= 1
                for v in W_new:
                    if count[v] == 0:
                        delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_d = d_new
            if best_d != d_old:
                # Apply the change.
                W_old = _dir_steps[d_old][x]
                W_new = _dir_steps[best_d][x]
                for v in W_old:
                    count[v] -= 1
                for v in W_new:
                    count[v] += 1
                d_x[x] = best_d
                current_N_size += best_delta
                improved = True
    
    # Return N (or S = complement).
    # The metric is |S| = 2401 - |N|.
    # We want to maximize |S|, so minimize |N|.
    # Return S as a bytes object.
    S = ~N
    return int(np.sum(S)), bytes(S)

def _valid_S(S):
    """Check that every point of F_q^2 is on a line with >=2 other S points."""
    # Wait, the validity condition is about N, not S.
    # "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N."
    # So we need to check N, not S.
    # But the function is called _valid_S, and it takes S.
    # We can just compute N = ~S and check N.
    N = ~S
    # For each x, check if there is a line through x such that L \ {x} subset N.
    for x in range(_SLOTS):
        ok = False
        for d in range(50):
            W = _dir_steps[d][x]
            if np.all(N[W]):
                ok = True
                break
        if not ok:
            return False
    return True

def _to_array(S):
    # N = complement of S
    pts = []
    for s in range(2401):
        if not S[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 560
    best_score = 0
    best_S = None
    seed = 12345
    it = 0
    while time.time() < deadline:
        score, Sb = _construct_local_search(seed)
        if score > best_score:
            best_score = score
            best_S = Sb
        seed += 1
        it += 1
        if it % 100 == 0:
            pass
    if best_S is None:
        # Fallback to a valid set (e.g., all points in N, S empty)
        S = np.zeros(_SLOTS, dtype=bool)
        best_S = bytes(S)
    return _to_array(bytearray(best_S))
