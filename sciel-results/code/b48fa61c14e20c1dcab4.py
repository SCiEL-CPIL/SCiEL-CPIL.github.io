# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using an optimized greedy algorithm.
    
    Strategy:
    1. Precompute lines and incidence.
    2. Use a greedy algorithm to build a large set S (complement of N).
    3. Optimized validity check:
       - Maintain C[x]: number of valid lines for x.
       - Maintain UniqueLine[x]: the index of the unique valid line if C[x]==1, else -1.
       - A point p can be added to S if it does not invalidate any x with C[x]==1.
         i.e., p is not on UniqueLine[x] for any x.
    4. Update C and UniqueLine after adding p.
    """
    p_prime = 7
    q = 49
    N_points = q * q  # 2401
    
    # --- Field Arithmetic F_49 ---
    def add_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        return ((a1 + a2) % 7) * 7 + (b1 + b2) % 7

    def mul_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        real = (a1 * a2 + 3 * b1 * b2) % 7
        imag = (a1 * b2 + b1 * a2) % 7
        return real * 7 + imag

    def inv_idx(i):
        a, b = divmod(i, 7)
        norm = (a * a - 3 * b * b) % 7
        if norm == 0:
            return 0
        norm_inv = pow(norm, -1, 7)
        real = (a * norm_inv) % 7
        imag = (6 * b * norm_inv) % 7
        return real * 7 + imag

    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    inv_table = np.zeros(49, dtype=np.int32)
    
    for i in range(49):
        for j in range(49):
            add_table[i, j] = add_idx(i, j)
            mul_table[i, j] = mul_idx(i, j)
        if i != 0:
            inv_table[i] = inv_idx(i)
        else:
            inv_table[i] = 0
            
    # --- Line Representation ---
    line_points = []
    
    # Vertical lines
    for x in range(49):
        pts = np.array([x * 49 + y for y in range(49)], dtype=np.int32)
        line_points.append(pts)
        
    # Sloped lines
    for m in range(49):
        for c in range(49):
            pts = np.array([x * 49 + add_idx(mul_table[m, x], c) for x in range(49)], dtype=np.int32)
            line_points.append(pts)
            
    # Precompute: for each point, list of line indices passing through it.
    point_lines = [[] for _ in range(2401)]
    for l_idx, pts in enumerate(line_points):
        for p in pts:
            point_lines[p].append(l_idx)
            
    # Convert point_lines to numpy arrays for faster iteration?
    # List of lists is fine.
    
    def greedy_build(seed):
        rng = np.random.RandomState(seed)
        order = np.arange(2401)
        rng.shuffle(order)
        
        in_S = np.zeros(2401, dtype=bool)
        C = np.full(2401, 50, dtype=np.int32)
        UniqueLine = np.full(2401, -1, dtype=np.int32)
        
        # Precompute a set of points with C[x] == 1 for fast checking?
        # We can just iterate over all x and check C[x]==1.
        # But we only need to check x that are on lines through p.
        # So we can just check those x.
        
        for p_idx in order:
            # Check if p_idx can be added
            # Condition: For all x on lines through p_idx, if C[x] == 1, then p_idx is NOT on UniqueLine[x].
            # Note: If x == p_idx, UniqueLine[x] is a line through p_idx.
            # If C[p_idx] == 1, and we add p_idx, does it invalidate p_idx?
            # If p_idx is added, the line L becomes dirty for p_idx?
            # No, L \ {p_idx} is disjoint from S.
            # So p_idx remains valid if C[p_idx] >= 1?
            # Wait.
            # If C[p_idx] == 1, the unique valid line is L*.
            # If we add p_idx to S, L* contains p_idx.
            # For p_idx, L* \ {p_idx} is disjoint from S?
            # Before adding, L* \ {p_idx} was disjoint from S (since L* was valid).
            # After adding, L* \ {p_idx} is still disjoint from S (since p_idx is removed from the check set).
            # So p_idx is NOT invalidated by adding itself.
            # So we only need to check x != p_idx.
            
            can_add = True
            # Get lines through p_idx
            lines_p = point_lines[p_idx]
            
            # Collect unique points on these lines (excluding p_idx)
            # We can just iterate and check.
            # To avoid double checking, we can use a set, but 50*49 is small.
            checked = set()
            
            for L in lines_p:
                for x in line_points[L]:
                    if x == p_idx:
                        continue
                    if x in checked:
                        continue
                    checked.add(x)
                    
                    if C[x] == 1:
                        # Check if L is the unique valid line for x?
                        # No, we need to check if p_idx is on UniqueLine[x].
                        # If UniqueLine[x] is a line through p_idx, then adding p_idx dirties it.
                        # So we check: is UniqueLine[x] in lines_p?
                        # Or simpler: is p_idx on UniqueLine[x]?
                        # Since we are iterating over lines through p_idx, if UniqueLine[x] is one of them,
                        # then p_idx is on it.
                        if UniqueLine[x] in lines_p:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                in_S[p_idx] = True
                # Update C and UniqueLine
                # For each line L in lines_p:
                #   L becomes dirty for all x in L, x != p_idx.
                #   So for those x, C[x] decreases by 1.
                #   If C[x] was 1, it becomes 0 (invalid).
                #   If C[x] was > 1, it decreases.
                #   If C[x] becomes 1, we need to find the new UniqueLine[x].
                
                for L in lines_p:
                    for x in line_points[L]:
                        if x == p_idx:
                            continue
                        # L was a valid line for x?
                        # It was valid if (L \ {x}) was disjoint from S.
                        # Before adding p_idx, was L valid for x?
                        # If p_idx was on L, and p_idx was not in S, then L \ {x} contained p_idx.
                        # So L was NOT valid for x if p_idx != x?
                        # Wait.
                        # L is valid for x if (L \ {x}) \cap S = empty.
                        # Before adding p_idx, p_idx was not in S.
                        # So if p_idx was on L \ {x}, L was already invalid for x?
                        # No, p_idx was NOT in S. So L \ {x} did not contain p_idx from S.
                        # So L WAS valid for x (assuming no other S points).
                        # After adding p_idx, p_idx IS in S.
                        # So L \ {x} contains p_idx (if x != p_idx).
                        # So L becomes INVALID for x.
                        
                        # So we decrement C[x] by 1 for all x on L, x != p_idx.
                        
                        old_C = C[x]
                        C[x] -= 1
                        
                        if old_C == 1:
                            # C[x] becomes 0. Invalid.
                            # UniqueLine[x] was L?
                            # We don't need to update UniqueLine if C[x] == 0.
                            pass
                        elif old_C == 2:
                            # C[x] becomes 1.
                            # We need to find the new unique valid line.
                            # Scan all lines through x.
                            # Find the one that is still valid.
                            # A line L' is valid if (L' \ {x}) \cap S = empty.
                            # We can check this by seeing if any point in L' \ {x} is in S.
                            # This is O(49) per line, O(50*49) per x.
                            # This is expensive if done often.
                            # But C[x] becomes 1 only once per point? No, it can go 50->49->...->1.
                            # So we do this 49 times per point? No, only when it hits 1.
                            # So once per point.
                            # Total 2401 * 2450 ops = 6M. OK.
                            
                            new_unique = -1
                            for L_prime in point_lines[x]:
                                # Check if L_prime is valid for x
                                # L_prime \ {x} disjoint from S?
                                valid = True
                                for y in line_points[L_prime]:
                                    if y == x:
                                        continue
                                    if in_S[y]:
                                        valid = False
                                        break
                                if valid:
                                    new_unique = L_prime
                                    break
                            UniqueLine[x] = new_unique
                        # If old_C > 2, C[x] >= 2, so UniqueLine[x] remains -1 (or irrelevant).
                        # If old_C == 1, C[x] == 0, UniqueLine[x] irrelevant.
                        
        return in_S

    start_time = time.time()
    deadline = start_time + 580
    
    best_in_S = None
    best_size = 0
    seed_id = 0
    
    while time.time() < deadline:
        in_S = greedy_build(seed_id)
        size = np.sum(in_S)
        if size > best_size:
            best_size = size
            best_in_S = in_S
        seed_id += 1
        
    if best_in_S is None:
        best_in_S = np.zeros(2401, dtype=bool)
        best_in_S[0] = True
        
    result_list = []
    for i in range(2401):
        if not best_in_S[i]:
            x = i // 49
            y = i % 49
            xa, xb = divmod(x, 7)
            ya, yb = divmod(y, 7)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
