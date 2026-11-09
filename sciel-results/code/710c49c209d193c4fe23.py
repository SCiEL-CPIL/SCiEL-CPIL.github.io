import time
import random
import itertools
import math

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute isosceles relations
# IsoPairsWithX[i]: list of (j, k) such that {i, j, k} is isosceles
# completion_map[i][j]: list of k such that {i, j, k} is isosceles
# Note: {i, j, k} is isosceles is symmetric.

IsoPairsWithX = [[] for _ in range(NUM_POINTS)]
# completion_map is a dict of (i, j) -> list of k
# We use a dictionary for sparse storage or a 2D list of lists
# 256x256 is small enough for a list of lists of lists?
# Let's use a dict for pairs to avoid 256x256 memory waste if sparse, 
# but 256x256 = 65k entries, each a list. It's fine.
completion_map = [[[] for _ in range(NUM_POINTS)] for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(j + 1, NUM_POINTS):
            if _is_iso(POINTS[i], POINTS[j], POINTS[k]):
                # Add to IsoPairsWithX
                IsoPairsWithX[i].append((j, k))
                IsoPairsWithX[j].append((i, k))
                IsoPairsWithX[k].append((i, j))
                
                # Add to completion_map
                completion_map[i][j].append(k)
                completion_map[i][k].append(j)
                completion_map[j][i].append(k)
                completion_map[j][k].append(i)
                completion_map[k][i].append(j)
                completion_map[k][j].append(i)

def _calculate_violations_iso(S_set):
    """Calculate number of isosceles triangles in S."""
    count = 0
    # To avoid double counting, we can iterate over pairs in S and check if the third point is in S?
    # Or just iterate over all triples? O(|S|^3) is too slow.
    # Use IsoPairsWithX.
    # For each i in S, count pairs (j, k) in IsoPairsWithX[i] such that j, k in S and j < k?
    # This counts each triangle 3 times.
    for i in S_set:
        for j, k in IsoPairsWithX[i]:
            if j in S_set and k in S_set:
                count += 1
    return count // 3

def _calculate_violations_max(S_set, BlockedCount):
    """Calculate number of unblocked points outside S."""
    count = 0
    for p in range(NUM_POINTS):
        if p not in S_set:
            if BlockedCount[p] == 0:
                count += 1
    return count

def _init_blocked_count(S_set):
    BlockedCount = [0] * NUM_POINTS
    for p in range(NUM_POINTS):
        if p in S_set:
            continue
        for a, b in IsoPairsWithX[p]:
            if a in S_set and b in S_set:
                BlockedCount[p] += 1
    return BlockedCount

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_solution = None
    best_size = float('inf')
    
    # Initial solution: Greedy maximal set
    # We want a small set, so maybe start with a random set and maximize?
    # Or start with a greedy independent set and then maximize?
    
    def _greedy_independent(order):
        S = []
        S_set = set()
        for p in order:
            # Check if p can be added without creating isosceles
            can_add = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if _is_iso(POINTS[p], POINTS[S[i]], POINTS[S[j]]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.append(p)
                S_set.add(p)
        return S, S_set

    def _make_maximal(S, S_set):
        # Add points greedily until maximal
        # We want to keep it small, so maybe don't add points?
        # But it MUST be maximal.
        # So we must add points if they can be added.
        # To minimize size, we should add points that "cover" the most unblocked points?
        # For now, just add any valid point.
        changed = True
        while changed:
            changed = False
            for p in range(NUM_POINTS):
                if p in S_set:
                    continue
                # Check if p can be added
                can_add = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if _is_iso(POINTS[p], POINTS[S[i]], POINTS[S[j]]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    S.append(p)
                    S_set.add(p)
                    changed = True
        return S, S_set

    # Generate initial solutions
    rng = random.Random(42)
    
    # 1. Greedy with random order
    for _ in range(20):
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        S, S_set = _greedy_independent(order)
        S, S_set = _make_maximal(S, S_set)
        
        # Check validity
        if _calculate_violations_iso(S_set) == 0:
            BlockedCount = _init_blocked_count(S_set)
            if _calculate_violations_max(S_set, BlockedCount) == 0:
                if len(S) < best_size:
                    best_size = len(S)
                    best_solution = list(S)
                    # print(f"Found valid set of size {best_size}")

    # 2. Simulated Annealing
    if best_solution is None:
        # Fallback
        S, S_set = _greedy_independent(list(range(NUM_POINTS)))
        S, S_set = _make_maximal(S, S_set)
        best_solution = list(S)
        best_size = len(S)

    # Start SA from best_solution or a random valid one
    # Let's try to improve the best solution found so far
    current_S = set(best_solution)
    current_List = list(best_solution)
    
    BlockedCount = _init_blocked_count(current_S)
    v_iso = _calculate_violations_iso(current_S)
    v_max = _calculate_violations_max(current_S, BlockedCount)
    
    C1 = 10000
    C2 = 10000
    C3 = 1
    
    energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
    
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute neighbors for fast moves?
    # We will use random moves.
    
    while time.time() < deadline and T > T_min:
        # Choose a move
        move_type = rng.random()
        if move_type < 0.3:
            # Add a point
            candidates = [p for p in range(NUM_POINTS) if p not in current_S]
            if not candidates:
                T *= cooling
                continue
            p = rng.choice(candidates)
            # Apply move
            current_S.add(p)
            
            # Update v_iso
            # Count new isosceles triangles formed with p
            new_v_iso = 0
            for a, b in IsoPairsWithX[p]:
                if a in current_S and b in current_S:
                    new_v_iso += 1
            v_iso += new_v_iso
            
            # Update BlockedCount
            # For each pair (p, b) with b in current_S
            # Find all k such that {p, b, k} is isosceles
            # For each such k, if k not in current_S, BlockedCount[k] += 1
            # Track changes in BlockedCount to update v_max
            v_max_delta = 0
            for b in current_S:
                if b == p: continue
                for k in completion_map[p][b]:
                    if k not in current_S:
                        if BlockedCount[k] == 0:
                            v_max_delta -= 1 # It was unblocked, now blocked
                        BlockedCount[k] += 1
            
            v_max += v_max_delta
            
            new_energy = C1 * v_iso + C2 * v_max + C3 * (len(current_S))
            delta_e = new_energy - energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                energy = new_energy
                current_List.append(p)
            else:
                # Revert
                current_S.remove(p)
                v_iso -= new_v_iso
                # Revert BlockedCount
                for b in current_S:
                    if b == p: continue
                    for k in completion_map[p][b]:
                        if k not in current_S:
                            if BlockedCount[k] == 1:
                                v_max_delta += 1 # It was blocked, now unblocked
                            BlockedCount[k] -= 1
                v_max += v_max_delta
                # Note: v_max_delta calculation on revert is tricky if we don't store state.
                # Better to recalculate v_max on revert? Or store old state.
                # For simplicity and safety, let's recalculate v_max if revert is frequent?
                # Or just accept the error? No, must be exact.
                # Let's just recalculate v_max and v_iso if revert happens?
                # That's expensive.
                # Let's store the old state of BlockedCount? No, too big.
                # Let's just do: if revert, recalculate v_iso and v_max from scratch?
                # O(N^2) is 256. It's fine.
                v_iso = _calculate_violations_iso(current_S)
                v_max = _calculate_violations_max(current_S, BlockedCount)
                # Wait, BlockedCount was reverted.
                # But v_max depends on BlockedCount.
                # So yes, recalculate v_max.
                energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
                
        elif move_type < 0.6:
            # Remove a point
            if not current_List:
                T *= cooling
                continue
            p = rng.choice(current_List)
            # Apply move
            current_S.remove(p)
            current_List.remove(p)
            
            # Update v_iso
            # Remove triangles containing p
            old_v_iso = v_iso
            # Count triangles containing p
            removed_v_iso = 0
            for a, b in IsoPairsWithX[p]:
                if a in current_S and b in current_S:
                    removed_v_iso += 1
            v_iso -= removed_v_iso
            
            # Update BlockedCount
            v_max_delta = 0
            for b in current_S:
                for k in completion_map[p][b]:
                    if k not in current_S:
                        if BlockedCount[k] == 1:
                            v_max_delta += 1 # Was blocked, now unblocked
                        BlockedCount[k] -= 1
            
            v_max += v_max_delta
            
            new_energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
            delta_e = new_energy - energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                energy = new_energy
            else:
                # Revert
                current_S.add(p)
                current_List.append(p)
                v_iso += removed_v_iso
                # Revert BlockedCount
                for b in current_S:
                    if b == p: continue
                    for k in completion_map[p][b]:
                        if k not in current_S:
                            if BlockedCount[k] == 0:
                                v_max_delta -= 1
                            BlockedCount[k] += 1
                v_max += v_max_delta
                # Recalculate to be safe
                v_iso = _calculate_violations_iso(current_S)
                v_max = _calculate_violations_max(current_S, BlockedCount)
                energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
                
        else:
            # Swap
            if not current_List:
                T *= cooling
                continue
            p_in = rng.choice(current_List)
            candidates_out = [p for p in range(NUM_POINTS) if p not in current_S]
            if not candidates_out:
                T *= cooling
                continue
            p_out = rng.choice(candidates_out)
            
            # Apply: remove p_in, add p_out
            current_S.remove(p_in)
            current_S.add(p_out)
            current_List.remove(p_in)
            current_List.append(p_out)
            
            # Recalculate v_iso and v_max from scratch for simplicity in swap
            # O(N^2) is acceptable for swap frequency
            v_iso = _calculate_violations_iso(current_S)
            BlockedCount = _init_blocked_count(current_S)
            v_max = _calculate_violations_max(current_S, BlockedCount)
            
            new_energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
            delta_e = new_energy - energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                energy = new_energy
                # Update best
                if v_iso == 0 and v_max == 0:
                    if len(current_S) < best_size:
                        best_size = len(current_S)
                        best_solution = list(current_S)
            else:
                # Revert
                current_S.remove(p_out)
                current_S.add(p_in)
                current_List.remove(p_out)
                current_List.append(p_in)
                v_iso = _calculate_violations_iso(current_S)
                BlockedCount = _init_blocked_count(current_S)
                v_max = _calculate_violations_max(current_S, BlockedCount)
                energy = C1 * v_iso + C2 * v_max + C3 * len(current_S)
                
        T *= cooling

    # Final check
    if best_solution is not None:
        # Verify
        S_set = set(best_solution)
        if _calculate_violations_iso(S_set) != 0 or _calculate_violations_max(S_set, _init_blocked_count(S_set)) != 0:
            # Fallback to a known valid greedy set
            S, S_set = _greedy_independent(list(range(NUM_POINTS)))
            S, S_set = _make_maximal(S, S_set)
            best_solution = list(S)
            
    return best_solution
