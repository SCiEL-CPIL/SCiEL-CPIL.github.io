import numpy as np
import time
import itertools
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        
        # Build incidence: for each triple, which 4-sets and 5-sets contain it
        # And for each 4-set/5-set, which triples it contains
        quad_triples = []  # list of 4 triple-indices per 4-set
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(quad[j] for j in range(4) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []  # list of 10 triple-indices per 5-set
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)
        
        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)
        
        # For each triple, the 4-sets and 5-sets it belongs to
        triple_to_4sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_4sets[t].append(qidx)
        triple_to_5sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_5sets[t].append(qidx)
        triple_to_4sets = [np.array(x, dtype=np.int32) for x in triple_to_4sets]
        triple_to_5sets = [np.array(x, dtype=np.int32) for x in triple_to_5sets]
        
        # Bit masks for fast violation checking
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        
        rng = np.random.RandomState(seed)
        
        # DPLL state
        # colors: -1 = unassigned, 0 = red, 1 = blue
        colors = np.full(M, -1, dtype=np.int8)
        
        # Bitsets for red and blue assigned triples
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        
        # Count assigned
        assigned_count = 0
        
        # Learned clauses: list of (triple_idx, forced_color) that must hold
        # Actually, let's use a simpler approach: track conflicts and use backtracking
        
        def count_violations():
            # Check red K4 violations
            r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
            # Check blue K5 violations
            b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
            return int(r4) + int(b5)
        
        def get_forced_assignments():
            """Find triples that are forced by current partial assignment.
            If 3 of 4 triples in a K4 are red, the 4th must be blue.
            If 9 of 10 triples in a K5 are blue, the 10th must be red.
            Returns list of (triple_idx, forced_color)"""
            forced = []
            
            # Check red K4: if 3 of 4 are red, 4th must be blue
            for qidx, tidxs in enumerate(quad_triples):
                red_count = 0
                unassigned = -1
                for t in tidxs:
                    if colors[t] == 0:
                        red_count += 1
                    elif colors[t] == -1:
                        unassigned = t
                if red_count == 3 and unassigned != -1 and colors[unassigned] == -1:
                    forced.append((unassigned, 1))  # must be blue
                elif red_count == 4:
                    # Already a violation - should have been caught
                    pass
            
            # Check blue K5: if 9 of 10 are blue, 10th must be red
            for qidx, tidxs in enumerate(quint_triples):
                blue_count = 0
                unassigned = -1
                for t in tidxs:
                    if colors[t] == 1:
                        blue_count += 1
                    elif colors[t] == -1:
                        unassigned = t
                if blue_count == 9 and unassigned != -1 and colors[unassigned] == -1:
                    forced.append((unassigned, 0))  # must be red
                elif blue_count == 10:
                    # Already a violation
                    pass
            
            return forced
        
        def apply_assignment(t, color):
            """Apply an assignment and update bitsets"""
            if colors[t] != -1:
                return False  # Already assigned
            
            colors[t] = color
            bit = np.uint64(1) << np.uint64(t & 63)
            w = t >> 6
            if color == 0:
                red_set[w] |= bit
            else:
                blue_set[w] |= bit
            return True
        
        def unapply_assignment(t):
            """Remove an assignment"""
            if colors[t] == -1:
                return
            color = colors[t]
            bit = np.uint64(1) << np.uint64(t & 63)
            w = t >> 6
            if color == 0:
                red_set[w] &= ~bit
            else:
                blue_set[w] &= ~bit
            colors[t] = -1
        
        # DPLL with propagation
        def dpll(depth=0):
            nonlocal assigned_count
            
            if time.time() > start_time + time_limit:
                return False
            
            # Check for violations
            if count_violations() > 0:
                return False
            
            # Check if all assigned
            if assigned_count == M:
                return True
            
            # Propagate forced assignments
            forced = get_forced_assignments()
            for t, color in forced:
                if colors[t] == -1:
                    if apply_assignment(t, color):
                        assigned_count += 1
                    else:
                        return False
            
            # Recheck after propagation
            if count_violations() > 0:
                return False
            
            if assigned_count == M:
                return True
            
            # Choose next unassigned triple
            # Heuristic: pick one that appears in most constraints
            unassigned = [t for t in range(M) if colors[t] == -1]
            if not unassigned:
                return count_violations() == 0
            
            # Pick the one with most constraints (4-sets + 5-sets)
            def constraint_count(t):
                return len(triple_to_4sets[t]) + len(triple_to_5sets[t])
            
            # For speed, just pick randomly from unassigned
            t = unassigned[rng.randint(len(unassigned))]
            
            # Try red first, then blue
            for color in [0, 1]:
                if apply_assignment(t, color):
                    assigned_count += 1
                    if dpll(depth + 1):
                        return True
                    unapply_assignment(t)
                    assigned_count -= 1
            
            return False
        
        # Initialize with a heuristic partial assignment
        # Use modular coloring as starting point
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1
            bit = np.uint64(1) << np.uint64(i & 63)
            w = i >> 6
            if colors[i] == 0:
                red_set[w] |= bit
            else:
                blue_set[w] |= bit
        assigned_count = M
        
        # First, check if the heuristic already works
        if count_violations() == 0:
            return colors.tolist()
        
        # Otherwise, use DPLL to fix violations
        # But DPLL from scratch is expensive. Instead, use a hybrid:
        # 1. Start from the heuristic
        # 2. Use targeted flips to eliminate violations (greedy)
        # 3. If stuck, use DPLL on a subset
        
        # Phase 1: Greedy violation elimination with propagation
        max_phase1_iter = 20000
        for it in range(max_phase1_iter):
            if time.time() > start_time + time_limit:
                break
            
            viol_count = count_violations()
            if viol_count == 0:
                return colors.tolist()
            
            # Find violating cliques
            viol_quads = np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]
            viol_quints = np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]
            
            # Collect candidate triples from violating cliques
            candidate_set = set()
            for q in viol_quads:
                for t in quad_triples[q]:
                    candidate_set.add(int(t))
            for q in viol_quints:
                for t in quint_triples[q]:
                    candidate_set.add(int(t))
            
            if not candidate_set:
                # No candidates - random perturbation
                t = int(rng.randint(0, M))
                old = colors[t]
                colors[t] = 1 - old
                bit = np.uint64(1) << np.uint64(t & 63)
                w = t >> 6
                if old == 0:
                    red_set[w] &= ~bit
                    blue_set[w] |= bit
                else:
                    blue_set[w] &= ~bit
                    red_set[w] |= bit
                continue
            
            # Score each candidate flip
            best_t = -1
            best_s = 0
            for t in candidate_set:
                old = colors[t]
                if old == 0:
                    # Flipping to blue: breaks red K4s, creates blue K5s
                    qmask = quad_masks[triple_to_4sets[t]]
                    broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                    qmask = quint_masks[triple_to_5sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                    s = int(broken) - int(created)
                else:
                    # Flipping to red: breaks blue K5s, creates red K4s
                    qmask = quint_masks[triple_to_5sets[t]]
                    broken = np.all((qmask & blue_set[None, :]) == qmask, axis=1).sum()
                    qmask = quad_masks[triple_to_4sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & red_set[None, :]) == without_t, axis=1).sum()
                    s = int(broken) - int(created)
                
                if s > best_s:
                    best_s = s
                    best_t = t
            
            if best_t == -1:
                # No single flip helps; random perturbation
                t = int(rng.randint(0, M))
                old = colors[t]
                colors[t] = 1 - old
                bit = np.uint64(1) << np.uint64(t & 63)
                w = t >> 6
                if old == 0:
                    red_set[w] &= ~bit
                    blue_set[w] |= bit
                else:
                    blue_set[w] &= ~bit
                    red_set[w] |= bit
                continue
            
            # Apply best flip
            t = best_t
            old = colors[t]
            colors[t] = 1 - old
            bit = np.uint64(1) << np.uint64(t & 63)
            w = t >> 6
            if old == 0:
                red_set[w] &= ~bit
                blue_set[w] |= bit
            else:
                blue_set[w] &= ~bit
                red_set[w] |= bit
        
        return None
    
    # Fallback valid small coloring
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    best_n = 9
    best_colors = fallback(9)
    
    # Try increasing n with multiple restarts
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        per_n_budget = min(remaining - 2.0, 30.0)
        if per_n_budget <= 0:
            break
        n_restarts = 3
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 13 + 7)
            if res is not None:
                found = res
                best_n = n
                best_colors = found
                break  # Found for this n, move to next n
        
    if best_n < 10:
        # Fallback to a known valid small coloring
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    return {'n': best_n, 'colors': best_colors}
