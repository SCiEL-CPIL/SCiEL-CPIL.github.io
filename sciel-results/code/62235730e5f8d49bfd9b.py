import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 148.0

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build(edges):
        m = len(edges)
        if m == 0:
            return 0, [], [], [], [], [], []
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        inc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].add(i)
            inc[v].add(i)
        # conflict_mask[i] is a bitmask of edges conflicting with edge i
        conflict_mask = [0] * m
        for i, (u, v) in enumerate(edges):
            forb = adj[u] | adj[v]
            mask = 0
            for w in forb:
                for j in inc[w]:
                    if j != i:
                        mask |= (1 << j)
            conflict_mask[i] = mask
        # Also precompute a list version for easier iteration if needed, but bitmask is faster for checks
        return n, adj, inc, conflict_mask, edges, [0] * m, [0] * m

    def greedy_order(order, conflict_mask, m):
        col = [-1] * m
        # used_mask[c] is a bitmask of edges already colored c
        # We can't have infinite colors easily in bitmask if m > 60, but m is ~160-240.
        # Python ints are arbitrary precision, so bitmask works for any m.
        # However, checking "is edge i compatible with color c" requires:
        # (conflict_mask[i] & used_mask[c]) == 0
        # This is very fast.
        used_mask = []
        for i in order:
            placed = False
            for c in range(len(used_mask)):
                if (conflict_mask[i] & used_mask[c]) == 0:
                    col[i] = c
                    used_mask[c] |= (1 << i)
                    placed = True
                    break
            if not placed:
                c = len(used_mask)
                col[i] = c
                used_mask.append(1 << i)
        return col

    def local_search_il(col, conflict_mask, m, deadline):
        # State: col list
        # We will use a simple ILS loop:
        # 1. Optimize current state via 1-moves and 2-swaps until local optimum
        # 2. Perturb
        # 3. Optimize again
        # Keep best found
        
        col = col[:]
        best_col = col[:]
        best_k = len(set(col))
        
        def get_classes(col):
            classes = defaultdict(list)
            for i, c in enumerate(col):
                if c >= 0:
                    classes[c].append(i)
            return classes

        def is_valid_col(col, conflict_mask):
            classes = get_classes(col)
            for c, es in classes.items():
                # Check pairwise conflicts within class
                # Optimization: for each edge e in class, check if it conflicts with any other in class
                # Using bitmask: for e in es, if (conflict_mask[e] & mask_of_class) != 0 (excluding self)
                # But building mask_of_class each time is O(|class|).
                # Simpler: nested loop with early exit, or bitmask check.
                # Given small class sizes, nested loop is fine, but let's use bitmask for speed.
                class_mask = 0
                for e in es:
                    class_mask |= (1 << e)
                for e in es:
                    # Check if e conflicts with any other in class
                    # Remove e from class mask
                    others = class_mask & ~(1 << e)
                    if (conflict_mask[e] & others) != 0:
                        return False
            return True

        def try_move_or_swap(col, conflict_mask, m):
            classes = get_classes(col)
            ckeys = list(classes.keys())
            if not ckeys:
                return False
            
            # Try 1-moves: move edge e from color a to color b
            # Condition: e does not conflict with any edge in color b
            # (conflict_mask[e] & mask_of_b) == 0
            
            # Precompute masks for each color class
            masks = {}
            for c in ckeys:
                mask = 0
                for e in classes[c]:
                    mask |= (1 << e)
                masks[c] = mask
            
            # 1-move
            for a in ckeys:
                for e in classes[a]:
                    for b in ckeys:
                        if b == a:
                            continue
                        if (conflict_mask[e] & masks[b]) == 0:
                            # Move e to b
                            col[e] = b
                            return True
            # 2-swap: swap e from a and f from b
            # Condition: e compatible with b\{f} AND f compatible with a\{e}
            # (conflict_mask[e] & (masks[b] & ~(1<<f))) == 0
            # (conflict_mask[f] & (masks[a] & ~(1<<e))) == 0
            
            for a in ckeys:
                for b in ckeys:
                    if b <= a:
                        continue
                    for e in classes[a]:
                        for f in classes[b]:
                            if (conflict_mask[e] & (masks[b] & ~(1 << f))) == 0 and \
                               (conflict_mask[f] & (masks[a] & ~(1 << e))) == 0:
                                # Swap
                                col[e] = b
                                col[f] = a
                                return True
            return False

        # Main ILS loop
        no_improve_count = 0
        temp = 1.0
        cooling = 0.995
        
        while time.time() < deadline:
            # Local optimization
            improved = True
            while improved:
                improved = False
                if try_move_or_swap(col, conflict_mask, m):
                    improved = True
                    # After a move, check if we reduced number of colors
                    # If a color class becomes empty, we might have reduced k
                    # We just recompute k at the end of local opt
            
            k = len(set(col))
            if k < best_k:
                best_k = k
                best_col = col[:]
                no_improve_count = 0
                # Reset temp if we found a new best? Optional.
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            
            # Perturbation
            # Pick a random edge and move it to a random compatible color, or a random 2-swap
            # To avoid getting stuck, we do a random move even if it increases k temporarily?
            # No, we must maintain validity.
            # Random 1-move or 2-swap
            classes = get_classes(col)
            ckeys = list(classes.keys())
            if not ckeys:
                break
            
            masks = {}
            for c in ckeys:
                mask = 0
                for e in classes[c]:
                    mask |= (1 << e)
                masks[c] = mask
            
            # Try to do a random valid move
            perturbed = False
            for _ in range(100):
                # Pick random edge
                e = random.randrange(m)
                a = col[e]
                # Pick random other color
                other_ckeys = [c for c in ckeys if c != a]
                if not other_ckeys:
                    continue
                b = random.choice(other_ckeys)
                if (conflict_mask[e] & masks[b]) == 0:
                    col[e] = b
                    perturbed = True
                    break
            
            if not perturbed:
                # Try random 2-swap
                for _ in range(100):
                    a = random.choice(ckeys)
                    b = random.choice(ckeys)
                    if a == b:
                        continue
                    if not classes[a] or not classes[b]:
                        continue
                    e = random.choice(classes[a])
                    f = random.choice(classes[b])
                    if (conflict_mask[e] & (masks[b] & ~(1 << f))) == 0 and \
                       (conflict_mask[f] & (masks[a] & ~(1 << e))) == 0:
                        col[e] = b
                        col[f] = a
                        perturbed = True
                        break
            
            if not perturbed:
                # Stuck? Try more aggressive perturbation: random re-coloring of a few edges
                # Or just break and restart from best
                break
                
            temp *= cooling

        if is_valid_col(best_col, conflict_mask):
            return best_col
        return col

    def solve_instance(edges, deadline):
        m = len(edges)
        n, adj, inc, conflict_mask, edges_list, _, _ = build(edges)
        
        if m == 0:
            return []

        # Initial solutions
        rng = random.Random(42 + m)
        orders = []
        # Degree-based order
        deg = [0] * n
        for u, v in edges_list:
            deg[u] += 1
            deg[v] += 1
        # Note: deg here is graph degree, not conflict degree.
        # Conflict degree is better.
        conf_deg = [bin(conflict_mask[i]).count('1') for i in range(m)]
        
        orders.append(sorted(range(m), key=lambda i: -conf_deg[i]))
        orders.append(sorted(range(m), key=lambda i: conf_deg[i]))
        orders.append(list(range(m)))
        
        # Random orders
        for _ in range(10):
            o = list(range(m))
            rng.shuffle(o)
            orders.append(o)

        candidates = []
        seen = set()
        for o in orders:
            col = greedy_order(o, conflict_mask, m)
            k = len(set(col))
            sig = tuple(col)
            if sig not in seen:
                seen.add(sig)
                candidates.append((k, col))
        
        candidates.sort(key=lambda x: x[0])
        
        # Take top 3 distinct candidates
        top = []
        tsig = set()
        for k, col in candidates:
            s = tuple(col)
            if s in tsig:
                continue
            tsig.add(s)
            top.append(col)
            if len(top) >= 3:
                break
        
        best_col = None
        best_k = 10**9
        
        # Allocate time for each candidate
        n_runs = len(top)
        time_per_run = (deadline - time.time()) / max(1, n_runs)
        
        for idx, col in enumerate(top):
            run_deadline = min(deadline - 0.5, time.time() + time_per_run)
            if time.time() > run_deadline:
                continue
            res = local_search_il(col, conflict_mask, m, run_deadline)
            rk = len(set(res))
            if rk < best_k:
                best_k = rk
                best_col = res
        
        if best_col is None:
            best_col = top[0]
        
        # Final validity check
        if not is_valid_col(best_col, conflict_mask):
             # Fallback to greedy
             best_col = greedy_order(list(range(m)), conflict_mask, m)
        
        # Remap colors to 0..k-1
        present = sorted(set(best_col))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[c] for c in best_col]

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    for idx, spec in enumerate(specs):
        edges = gen_instance(*spec)
        m = len(edges)
        left_after = len(specs) - 1 - idx
        # Allocate time based on remaining instances
        # Give more time to later instances? Or equal?
        # Let's give equal time, but ensure we don't exceed deadline.
        inst_dl = min(DEADLINE - 2.0 - 2.0 * left_after, time.time() + 48.0)
        if inst_dl < time.time() + 5.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 5.0)
        
        try:
            out.append(solve_instance(edges, inst_dl))
        except Exception:
            # Fallback
            n, adj, inc, conflict_mask, edges_list, _, _ = build(edges)
            col = greedy_order(list(range(m)), conflict_mask, m)
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    
    return out
