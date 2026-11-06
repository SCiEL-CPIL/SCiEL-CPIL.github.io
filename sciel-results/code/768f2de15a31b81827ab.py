import sys
sys.setrecursionlimit(10000)

def construct():
    N = 256
    
    # Precompute 3-APs and 4-APs
    # triples_3[x] = list of (y, z) such that {x, y, z} is a 3-AP
    # quads_4[x] = list of (y, z, w) such that {x, y, z, w} is a 4-AP
    
    triples_3 = [[] for _ in range(N + 1)]
    quads_4 = [[] for _ in range(N + 1)]
    
    # Find all 3-APs
    # a < b < c, b = (a+c)/2
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N:
                break
            if c > b:
                # {a, b, c} is a 3-AP
                triples_3[a].append((b, c))
                triples_3[b].append((a, c))
                triples_3[c].append((a, b))
                
    # Find all 4-APs
    # a < b < c < d, b-a = c-b = d-c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            c = b + d
            if c > N:
                break
            e = c + d
            if e > N:
                break
            # {a, b, c, e} is a 4-AP
            quads_4[a].append((b, c, e))
            quads_4[b].append((a, c, e))
            quads_4[c].append((a, b, e))
            quads_4[e].append((a, b, c))
            
    # Initial greedy solution to set a good lower bound
    def get_greedy_solution():
        selected = set()
        for x in range(1, N + 1):
            # Check if adding x creates a 4-AP
            valid = True
            for (y, z, w) in quads_4[x]:
                if y in selected and z in selected and w in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected

    initial_selected = get_greedy_solution()
    
    # Count 3-APs in initial solution
    def count_3aps_in_set(S):
        count = 0
        S_list = sorted(S)
        S_set = set(S_list)
        for a in S_list:
            for b in S_list:
                if b <= a: continue
                c = 2 * b - a
                if c in S_set and c > b:
                    count += 1
        return count

    best_score = count_3aps_in_set(initial_selected)
    best_set = initial_selected

    # Helper for B&B
    # selected: set of currently selected elements
    # index: next element to consider (1-based)
    # current_score: number of 3-APs in 'selected'
    # potential_ub: upper bound on total 3-APs (count_3APs in selected U {index..N})
    
    # Precompute total 3-APs in any subset of [k, N]?
    # Actually, we maintain potential_ub incrementally.
    # Initial potential_ub = total 3-APs in [1, N]
    total_3APs_all = 0
    for x in range(1, N + 1):
        total_3APs_all += len(triples_3[x]) // 3 # Each 3-AP counted 3 times
    
    # We need to track which elements are in 'selected' for fast lookup
    # And we need to calculate N_x (3-APs involving x and elements in selected U {x+1..N})
    
    # To speed up N_x calculation:
    # N_x = count of (y, z) in triples_3[x] such that (y in selected OR y > x) AND (z in selected OR z > x)
    
    def solve(selected, index, current_score, potential_ub):
        nonlocal best_score, best_set
        
        if index > N:
            if current_score > best_score:
                best_score = current_score
                best_set = set(selected)
            return
            
        # Prune
        if potential_ub < best_score:
            return
            
        x = index
        
        # --- Branch 1: Exclude x ---
        # Calculate N_x: 3-APs in (selected U {x+1..N}) that involve x
        # Elements available for pairs: selected (indices < x) and {x+1..N} (indices > x)
        # Note: elements in [1..x-1] that are NOT in selected are not in the set.
        
        n_x = 0
        for (y, z) in triples_3[x]:
            # y and z must be in (selected U {x+1..N})
            # Condition for v: v in selected OR v > x
            # Note: v cannot be x.
            if (y in selected or y > x) and (z in selected or z > x):
                n_x += 1
                
        new_potential_ub = potential_ub - n_x
        
        # Recurse Exclude
        solve(selected, index + 1, current_score, new_potential_ub)
        
        # --- Branch 2: Include x ---
        # Check 4-AP constraint
        valid = True
        for (y, z, w) in quads_4[x]:
            if y in selected and z in selected and w in selected:
                valid = False
                break
                
        if valid:
            # Calculate new 3-APs formed by x and two elements in selected
            gain = 0
            for (y, z) in triples_3[x]:
                if y in selected and z in selected:
                    gain += 1
                    
            new_selected = selected | {x}
            new_score = current_score + gain
            # potential_ub remains the same because the set of elements (selected U {x+1..N}) 
            # becomes (selected U {x} U {x+1..N}) which is the same set of numbers.
            solve(new_selected, index + 1, new_score, potential_ub)
            
    # Start B&B
    # Initial state: selected={}, index=1, current_score=0, potential_ub=total_3APs_all
    # But we can start with the greedy solution's prefix? No, B&B explores all.
    # To make it faster, we can order branches. Usually, including high-value nodes first finds good solutions.
    # Here, we don't have a heuristic for "value" of x easily.
    # We can try to include x first if it has high connectivity?
    # For now, standard order.
    
    # Optimization: Use a list for selected to avoid set overhead? 
    # Set is O(1) lookup. List is O(N). N=256, so List is fine too, but Set is cleaner.
    # However, creating new sets at every node is expensive.
    # We can use a boolean array `in_selected` and backtrack.
    
    # Let's rewrite with backtracking to avoid memory overhead.
    
    in_selected = [False] * (N + 1)
    
    def solve_bt(index, current_score, potential_ub):
        nonlocal best_score, best_set
        
        if index > N:
            if current_score > best_score:
                # Construct set
                s = set()
                for i in range(1, N + 1):
                    if in_selected[i]:
                        s.add(i)
                best_set = s
                best_score = current_score
            return
            
        if potential_ub < best_score:
            return
            
        x = index
        
        # --- Branch 1: Exclude x ---
        n_x = 0
        for (y, z) in triples_3[x]:
            # y, z in (selected U {x+1..N})
            # selected: in_selected[v] is True
            # future: v > x
            if (in_selected[y] or y > x) and (in_selected[z] or z > x):
                n_x += 1
                
        new_potential_ub = potential_ub - n_x
        
        solve_bt(index + 1, current_score, new_potential_ub)
        
        # --- Branch 2: Include x ---
        valid = True
        for (y, z, w) in quads_4[x]:
            if in_selected[y] and in_selected[z] and in_selected[w]:
                valid = False
                break
                
        if valid:
            gain = 0
            for (y, z) in triples_3[x]:
                if in_selected[y] and in_selected[z]:
                    gain += 1
                    
            in_selected[x] = True
            solve_bt(index + 1, current_score + gain, potential_ub)
            in_selected[x] = False
            
    solve_bt(1, 0, total_3APs_all)
    
    return sorted(best_set)

# Run construct to verify
# res = construct()
# print(len(res))
# print(res)

if __name__ == "__main__":
    construct()
