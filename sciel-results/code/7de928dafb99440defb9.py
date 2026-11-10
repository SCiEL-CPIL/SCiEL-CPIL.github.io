import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

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

    # Build all lines
    # Non-vertical: m in 0..48, c in 0..48 -> line_id = m*49 + c (total 2401)
    # Vertical: x0 in 0..48 -> line_id = 2401 + x0 (total 49)
    # Total: 2450 lines
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

    line_points_list = [list(s) for s in line_points]
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for small sizes
    # Let's stick to lists for flexibility, but ensure they are optimized.

    def init_parabola():
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

    def init_random(density=0.02):
        in_C = np.zeros(q2, dtype=bool)
        n_points = int(q2 * density)
        indices = random.sample(range(q2), n_points)
        for i in indices:
            in_C[i] = True
        return in_C

    def compute_witness(in_C):
        """
        For each point i, find all lines L containing i such that
        no other point in L is in C.
        """
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

    def is_valid(in_C, W):
        for i in range(q2):
            if len(W[i]) == 0:
                return False
        return True

    def greedy_add(in_C, W, rng, max_steps=20000):
        """
        Greedy add points to C while maintaining validity.
        Uses a priority queue to efficiently select points that cause minimal disruption.
        """
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = np.array([len(s) for s in W], dtype=np.int32)

        if not is_valid(in_C, W):
            return in_C, W, wcount, 0

        candidates = [i for i in range(q2) if not in_C[i]]
        added = 0

        # Priority queue: (cost, point_index)
        # Cost: number of points that would lose their only witness if this point is added.
        # Actually, simpler cost: number of witness lines invalidated.
        # Let's use the "danger" metric: how many points have wcount==1 and share a line with this point?
        
        # Initialize heap
        heap = []
        for i in candidates:
            # Calculate initial danger
            danger = 0
            for lid in point_lines[i]:
                for x_idx in line_points_list[lid]:
                    if x_idx == i:
                        continue
                    if lid in W[x_idx] and wcount[x_idx] == 1:
                        danger += 1
            heapq.heappush(heap, (danger, i))

        for _ in range(max_steps):
            if time.time() > _DEADLINE - _MARGIN:
                break
            if not heap:
                break
            
            danger, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
            
            # Verify danger is still accurate (Lazy Update)
            current_danger = 0
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx == p_idx:
                        continue
                    if lid in W[x_idx] and wcount[x_idx] == 1:
                        current_danger += 1
            
            if current_danger != danger:
                heapq.heappush(heap, (current_danger, p_idx))
                continue

            if current_danger > 0:
                # If danger is not 0, adding this point would make some point invalid.
                # We could try to repair, but for now, skip.
                # Or we can break if no zero-danger points remain.
                # Since heap is sorted, if min danger > 0, all are > 0.
                break

            # Safe to add
            in_C[p_idx] = True
            added += 1
            
            # Update W and wcount for neighbors
            # Also update heap for neighbors
            neighbors_to_update = set()
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx == p_idx:
                        continue
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
                        if not in_C[x_idx]:
                            neighbors_to_update.add(x_idx)
            
            for x_idx in neighbors_to_update:
                # Recalculate danger for x_idx
                new_danger = 0
                for lid2 in point_lines[x_idx]:
                    for y_idx in line_points_list[lid2]:
                        if y_idx == x_idx:
                            continue
                        if lid2 in W[y_idx] and wcount[y_idx] == 1:
                            new_danger += 1
                heapq.heappush(heap, (new_danger, x_idx))

        return in_C, W, wcount, added

    def local_search(in_C, W, wcount, rng, max_iters=50000):
        """
        Local search: try to add points to C.
        This is similar to greedy_add but allows random sampling if greedy gets stuck?
        Actually, greedy_add with heap already covers this well.
        Let's add a simple random local search that tries to swap or add if possible.
        """
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        candidates = [i for i in range(q2) if not in_C[i]]
        improved = 0
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - _MARGIN:
                break
            if not candidates:
                break
            
            # Pick random candidate
            p_idx = rng.choice(candidates)
            
            # Check if adding p_idx is safe
            safe = True
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx == p_idx:
                        continue
                    if lid in W[x_idx] and wcount[x_idx] == 1:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                in_C[p_idx] = True
                improved += 1
                candidates.remove(p_idx)
                
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx == p_idx:
                            continue
                        if lid in W[x_idx]:
                            W[x_idx].discard(lid)
                            wcount[x_idx] -= 1
        
        return in_C, W, wcount, improved

    def repair(in_C):
        """
        Try to repair an invalid C by removing points until valid.
        """
        in_C = in_C.copy()
        W = compute_witness(in_C)
        
        while not is_valid(in_C, W):
            # Find a point with no witness
            bad_points = [i for i in range(q2) if len(W[i]) == 0]
            if not bad_points:
                break
            
            # Remove a random point from C that might help
            c_points = [i for i in range(q2) if in_C[i]]
            if not c_points:
                break
            
            # Try removing each C point and check if it helps
            removed = False
            # Shuffle to avoid bias
            rng_local = random.Random(time.time_ns())
            rng_local.shuffle(c_points)
            
            for p_idx in c_points:
                in_C[p_idx] = False
                W_new = compute_witness(in_C)
                if is_valid(in_C, W_new):
                    W = W_new
                    removed = True
                    break
                else:
                    in_C[p_idx] = True
            
            if not removed:
                break
        
        W = compute_witness(in_C)
        return in_C, W, is_valid(in_C, W)

    best_in_C = None
    best_size = 0
    
    rng = random.Random(42)
    
    # Try different initial configurations
    # The key insight from the prompt is that starting with a larger set (like a unital or dense random) 
    # and repairing it can yield better results than starting small and adding.
    # However, the current code starts with a parabola (small) and adds.
    # Let's try a hybrid: Start with a moderately dense random set, repair it, then greedily add more.
    
    initializers = [
        ('parabola', init_parabola),
        ('random_001', lambda: init_random(0.001)),
        ('random_005', lambda: init_random(0.005)),
        ('random_01', lambda: init_random(0.01)),
        ('random_02', lambda: init_random(0.02)),
        ('random_05', lambda: init_random(0.05)), # Denser start
    ]
    
    for trial in range(30):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Cycle through initializers
        name, init_fn = initializers[trial % len(initializers)]
        in_C = init_fn()
        
        # Check if valid
        W = compute_witness(in_C)
        if not is_valid(in_C, W):
            in_C, W, valid = repair(in_C)
            if not valid:
                continue
        
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        # Greedy add
        in_C, W, wcount, added1 = greedy_add(in_C, W, rng, max_steps=10000)
        
        # Local search
        in_C, W, wcount, added2 = local_search(in_C, W, wcount, rng, max_iters=30000)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
    
    # If no valid solution found, fall back to parabola
    if best_in_C is None:
        best_in_C = init_parabola()
    
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
