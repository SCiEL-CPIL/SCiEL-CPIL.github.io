import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    # Precompute: for each point, which lines pass through it (already in point_lines)
    # Precompute: for each line, the list of points on it (convert sets to lists for speed)
    line_points_list = [list(s) for s in line_points]
    
    def init_C_parabola():
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C
    
    def compute_W(in_C):
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        return W
    
    def is_valid_Nikodym(in_C):
        """Check that every point has at least one witness line."""
        for i in range(q2):
            if not any(
                not (line_points_list[lid] and any(q_idx != i and in_C[q_idx] for q_idx in line_points_list[lid]))
                for lid in point_lines[i]
            ):
                return False
        return True
    
    def try_greedy_add(in_C, W, rng, max_steps=5000):
        """Greedy add points to C, using witness counts for efficiency."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        
        # witness_count[x] = len(W[x])
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        # For each point p not in C, compute how many points it would affect
        candidates = [i for i in range(q2) if not in_C[i]]
        
        # Build a heap: (damage, point_idx) where damage = number of points that would lose a witness
        # Actually, we need: for point p, the set of x such that p is on a witness line of x.
        # When we add p, those x lose that witness line. If wcount[x] becomes 0, it's invalid.
        
        # Precompute: for each point p, the list of (x, lid) such that lid in W[x] and p in line_points[lid] and p != x
        # This is expensive, so we compute on the fly.
        
        steps = 0
        while steps < max_steps and time.time() < _DEADLINE - _MARGIN:
            steps += 1
            
            # Find the point p that minimizes damage
            best_p = -1
            best_damage = float('inf')
            
            # Sample a subset of candidates for speed
            sample_size = min(len(candidates), 200)
            if sample_size < len(candidates):
                sample = rng.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for p_idx in sample:
                if in_C[p_idx]:
                    continue
                # Compute damage: number of x that have a witness line through p_idx
                # and would have wcount drop to 0
                damage = 0
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx == p_idx:
                            continue
                        if lid in W[x_idx]:
                            if wcount[x_idx] == 1:
                                damage += 1
                if damage < best_damage:
                    best_damage = damage
                    best_p = p_idx
            
            if best_p == -1 or best_damage > 0:
                # Try to add the best one even if damage > 0? No, must be 0.
                # If no zero-damage point found in sample, break or try more
                if best_damage > 0:
                    break
            
            # Add best_p to C
            in_C[best_p] = True
            if best_p in candidates:
                candidates.remove(best_p)
            
            # Update W and wcount
            for lid in point_lines[best_p]:
                for x_idx in line_points_list[lid]:
                    if x_idx == best_p:
                        continue
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
        
        return in_C, W, wcount
    
    def local_search_remove(in_C, W, wcount, rng, max_iters=10000):
        """Try to remove points from C (add to N) while maintaining validity."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        c_points = [i for i in range(q2) if in_C[i]]
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - _MARGIN:
                break
            if not c_points:
                break
            
            # Pick a random point from C to try removing
            idx = rng.randrange(len(c_points))
            p_idx = c_points[idx]
            
            # Try removing p_idx: for each line through p_idx, check if removing p_idx
            # would cause any point x to lose its last witness line
            # A point x is affected if there's a line L through p_idx such that L in W[x] and p_idx != x
            # After removal, L is still a valid witness for x (since we're removing from C, not adding)
            # Wait: removing p_idx from C means p_idx is no longer in C.
            # For a point x, a line L through x is a witness if (L \ {x}) ∩ C = ∅.
            # If we remove p_idx from C, then any line L that contained p_idx (and p_idx != x)
            # becomes a valid witness for x if no other point of C is on L.
            # So removing from C can only help, not hurt!
            # Therefore, we can always remove any point from C and the Nikodym property is preserved.
            # Wait, that can't be right. Let me re-examine.
            
            # Actually: N is the Nikodym set. C = complement. The condition is:
            # For every x, there exists a line L through x such that (L \ {x}) ⊆ N,
            # i.e., (L \ {x}) ∩ C = ∅.
            # If we remove a point p from C (so p joins N), then for any x, any line L through x
            # that previously had p in (L \ {x}) ∩ C now has p not in C, so L might become valid.
            # So removing from C can only make the condition easier to satisfy.
            # Therefore, ANY subset of C is also a valid C!
            # That means the maximal C is the entire plane, and N = ∅, which is trivially valid?
            # But N must have k >= 1. And the metric is 2401 - |N| = |C|, so we want C as large as possible.
            # If any subset of C is valid, then C = F_49^2 is valid, and N = ∅.
            # But the task says k >= 1, so N must be non-empty.
            
            # Wait, I think I have the direction wrong. Let me re-read.
            # "N must be a Nikodym set: every point x of F_49^2 (in N or not) must lie on a line
            # {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1),
            # whose points other than x all belong to N."
            # So for every x, there's a line through x where all OTHER points are in N.
            # This means C = F_49^2 \ N must be such that for every x, there's a line through x
            # that contains no point of C other than possibly x itself.
            # If x ∈ C, the line can contain x (which is in C) but no other point of C.
            # If x ∉ C, the line must contain no point of C at all.
            
            # So C can be any set where no line contains two or more points of C, UNLESS...
            # Wait: if x ∈ C, the line through x can contain x but no other C point.
            # If x ∉ C, the line through x must contain no C points at all.
            # This is equivalent to: for every line L, |L ∩ C| ≤ 1.
            # Because if a line L has two points p, q in C, then for x = p, any line through p
            # that contains q would have q in C and q != p, violating the condition for x = p
            # on that line. But p might have another line that avoids C. So it's not that no line
            # can have two C points; it's that for each point x, there must exist at least one line
            # through x with at most one C point (itself if x in C, or zero if x not in C).
            
            # So removing from C is always safe. The maximal valid C is the largest set with the property
            # that every point has a line through it avoiding C (except itself).
            # This is equivalent to: C is a set such that the "visibility" graph is connected in a certain way.
            
            # Actually, I realize the current code's approach is correct: it starts with a small C and
            # greedily adds points. The constraint is that after adding, every point still has a witness line.
            # Removing from C is always safe, so the local search should try to remove points from C
            # to... wait, that would make C smaller, which is worse for the metric.
            
            # I'm confused. The metric is 2401 - |N| = |C|. We want to MAXIMIZE |C|.
            # The current code starts with C = parabola (49 points) and adds points to C.
            # This is correct: we want to grow C.
            # Removing from C would shrink C, which is bad.
            # So local search should not remove from C; it should try to add more points.
            
            break
        
        return in_C, W, wcount
    
    best_in_C = init_C_parabola()
    best_size = int(np.sum(best_in_C))
    
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        in_C = init_C_parabola()
        W = compute_W(in_C)
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        in_C, W, wcount = try_greedy_add(in_C, W, rng, max_steps=10000)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
