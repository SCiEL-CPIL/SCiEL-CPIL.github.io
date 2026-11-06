"""
Constraint Satisfaction approach for minimum color 4-AP coloring.
Uses backtracking with dynamic variable ordering and value ordering.
"""
import time
import random
import sys

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    N = 128
    
    # Precompute all non-constant 4-term APs (a, b, c, d) with a < b < c < d and b-a = c-b = d-c
    ap_list = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))
    
    # Build adjacency: for each position, list of APs it's involved in
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    # For each AP, store the 4 positions
    ap_positions = [(a, b, c, e) for (a, b, c, e) in ap_list]
    
    def is_valid_assignment(colors, pos, color):
        """Check if assigning 'color' to 'pos' violates any constraint with already-assigned positions."""
        for ap_idx in pos_aps[pos]:
            a, b, c, e = ap_positions[ap_idx]
            # Get colors of other 3 positions
            others = []
            for p in (a, b, c, e):
                if p != pos:
                    if colors[p] == -1:
                        continue  # Not assigned yet
                    others.append(colors[p])
            
            if len(others) == 3:
                # All other 3 are assigned
                all_colors = others + [color]
                if len(set(all_colors)) < 3:
                    return False
            elif len(others) == 2:
                # 2 others assigned, 1 unassigned
                # The unassigned one can be anything, so we just need that
                # the 2 assigned + color can potentially reach 3 colors
                # Actually, we need to check: is there ANY value for the unassigned
                # that makes the total have >= 3 colors?
                # The unassigned can be any color, so as long as {color} ∪ others has potential
                # to reach 3, it's OK. Since unassigned can be new, we just need
                # that the constraint is not already violated by the assigned ones forcing < 3.
                # If others + color has 1 or 2 distinct colors, the unassigned can add a 3rd.
                # So this is always satisfiable unless... actually it's always satisfiable
                # because the unassigned position can choose a new color.
                pass
            # If 0 or 1 others assigned, definitely satisfiable
        return True
    
    def get_unassigned_positions(colors):
        """Get list of unassigned positions."""
        return [i for i in range(N) if colors[i] == -1]
    
    def count_assigned_colors(colors):
        """Count distinct colors used so far."""
        used = set(colors[i] for i in range(N) if colors[i] != -1)
        return len(used)
    
    def backtracking_solve(K, max_depth=2000, timeout=None):
        """
        Backtracking solver with dynamic variable ordering.
        Tries to find a valid K-coloring.
        """
        if timeout is None:
            timeout = deadline - time.time()
        
        colors = [-1] * N
        rng = random.Random(12345)
        
        # We'll use a recursive approach but with iterative deepening on K
        # First, try to build a solution greedily with backtracking
        
        def solve_recursive(depth, colors, used_colors):
            if time.time() > deadline:
                return False
            
            if depth == N:
                return True
            
            # Dynamic variable ordering: pick the most constrained unassigned position
            # Heuristic: position with most assigned neighbors in APs
            unassigned = [i for i in range(N) if colors[i] == -1]
            if not unassigned:
                return True
            
            # Score each unassigned position by how many APs have 3+ assigned
            def constraint_score(pos):
                score = 0
                for ap_idx in pos_aps[pos]:
                    a, b, c, e = ap_positions[ap_idx]
                    assigned_count = 0
                    for p in (a, b, c, e):
                        if p != pos and colors[p] != -1:
                            assigned_count += 1
                    if assigned_count >= 2:
                        score += 10
                    elif assigned_count >= 1:
                        score += 1
                return score
            
            # Pick position with highest constraint score
            best_pos = max(unassigned, key=constraint_score)
            
            # Value ordering: try colors already in use first, then new colors
            values = list(used_colors)
            for c in range(K):
                if c not in used_colors:
                    values.append(c)
            
            for color in values:
                if not is_valid_assignment(colors, best_pos, color):
                    continue
                
                colors[best_pos] = color
                new_used = set(used_colors)
                new_used.add(color)
                
                if solve_recursive(depth + 1, colors, new_used):
                    return True
                
                colors[best_pos] = -1
            
            return False
        
        # Try with initial K colors
        result = solve_recursive(0, colors, set())
        if result:
            return colors
        
        return None
    
    # Strategy: Try K=3, then K=4, etc.
    # But backtracking might be too slow. Let's use a hybrid approach.
    
    # First, try a smarter greedy with lookahead
    def greedy_with_backtrack(K, max_backtracks=5000):
        """
        Greedy assignment with backtracking when stuck.
        """
        colors = [-1] * N
        order = list(range(N))
        rng = random.Random(99)
        
        backtracks = 0
        
        def try_assign(pos, colors, depth):
            nonlocal backtracks
            if time.time() > deadline:
                return False, None
            
            if depth == N:
                return True, colors[:]
            
            # Pick next position using dynamic ordering
            unassigned = [i for i in range(N) if colors[i] == -1]
            if not unassigned:
                return True, colors[:]
            
            # Most constrained first
            def c_score(p):
                s = 0
                for ap_idx in pos_aps[p]:
                    a, b, c, e = ap_positions[ap_idx]
                    ac = sum(1 for x in (a,b,c,e) if x != p and colors[x] != -1)
                    if ac >= 2: s += 5
                    elif ac >= 1: s += 1
                return s
            
            pos = max(unassigned, key=c_score)
            
            # Try colors in order: existing colors first
            used = set(colors[i] for i in range(N) if colors[i] != -1)
            candidates = list(used) + [c for c in range(K) if c not in used]
            rng.shuffle(candidates)  # Add randomness to break symmetry
            
            for color in candidates:
                if not is_valid_assignment(colors, pos, color):
                    continue
                
                colors[pos] = color
                success, result = try_assign(pos, colors, depth + 1)
                if success:
                    return True, result
                colors[pos] = -1
                
                backtracks += 1
                if backtracks > max_backtracks:
                    return False, None
            
            return False, None
        
        success, result = try_assign(0, colors, 0)
        if success and result:
            return result
        return None
    
    # Try K=3
    for K in [3, 4, 5]:
        result = greedy_with_backtrack(K, max_backtracks=10000)
        if result is not None:
            # Verify
            valid = True
            for a, b, c, e in ap_list:
                if len(set([result[a], result[b], result[c], result[e]])) < 3:
                    valid = False
                    break
            if valid:
                # Relabel to consecutive
                mapping = {}
                new_res = []
                next_id = 0
                for c in result:
                    if c not in mapping:
                        mapping[c] = next_id
                        next_id += 1
                    new_res.append(mapping[c])
                return new_res
    
    # Fallback: Use a known good pattern or random search
    # Try simulated annealing with better initialization
    def simulated_annealing(K, max_iters=100000):
        rng = random.Random(42)
        colors = [rng.randint(0, K - 1) for _ in range(N)]
        
        def count_violations():
            count = 0
            for a, b, c, e in ap_list:
                ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
                if ca == cb:
                    if cc == ce:
                        count += 1
                    else:
                        if ca == cc or ca == ce:
                            count += 1
                else:
                    if cc == ce:
                        if cc == ca or cc == cb:
                            count += 1
                    else:
                        if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                            count += 1
            return count
        
        current_cost = count_violations()
        if current_cost == 0:
            return colors
        
        best_cost = current_cost
        best_colors = colors[:]
        T = 1.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            pos = rng.randint(0, N - 1)
            old_color = colors[pos]
            new_color = rng.randint(0, K - 1)
            if new_color == old_color:
                new_color = (new_color + 1) % K
            
            colors[pos] = new_color
            new_cost = count_violations()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    return colors
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                delta = new_cost - current_cost
                if rng.random() < pow(2.718, -delta / T):
                    current_cost = new_cost
                else:
                    colors[pos] = old_color
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        if best_cost == 0:
            return best_colors
        return None
    
    for K in [3, 4]:
        result = simulated_annealing(K)
        if result is not None:
            mapping = {}
            new_res = []
            next_id = 0
            for c in result:
                if c not in mapping:
                    mapping[c] = next_id
                    next_id += 1
                new_res.append(mapping[c])
            return new_res
    
    # Ultimate fallback: 128 unique colors
    return list(range(N))
