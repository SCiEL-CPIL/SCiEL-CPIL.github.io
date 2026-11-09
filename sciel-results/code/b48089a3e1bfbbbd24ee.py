import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 150s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Optimized Bidirectional growth with Warnsdorff's rule.
    Uses bitmask-based used set for speed.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def bidirectional_grow(start_vertex=0, time_limit=None):
        """
        Grow a snake bidirectionally with optimized checks using forbidden bitmask.
        """
        v0 = start_vertex
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        # forbidden[v] = bitmask of vertices that are adjacent to v (i.e., forbidden for use)
        # We maintain a global "blocked" set: a vertex is blocked if it's used or adjacent to a used vertex
        # Actually, we need: a candidate c is valid if it's not used AND not adjacent to any used vertex
        # EXCEPT the end vertex it connects to.
        # Let's maintain: adj_mask[v] = set of neighbors of v
        # blocked[v] = True if v is used OR v is adjacent to any used vertex
        # But we need to exclude the specific end vertex's adjacency.
        
        # Simpler: maintain a count array. For each vertex, count how many used vertices are adjacent to it.
        # A vertex c is valid to add to end e if:
        #   not used[c] AND (adj_count[c] == 1)  # only adjacent to the end e
        # Wait, adj_count counts ALL used neighbors. If c is adjacent to end e, and no other used vertex,
        # then adj_count[c] should be 1 (just e). But we haven't added c yet, so e is in used.
        # Actually, let me think again.
        
        # When we consider adding candidate c to end e (which is in path):
        # c must not be used.
        # c must not be adjacent to any p in path EXCEPT e.
        # Since e is in path and c is adjacent to e (by construction, c is a neighbor of e),
        # the condition is: c is not adjacent to any p in path where p != e.
        # Equivalently: the number of used vertices adjacent to c is exactly 1 (just e).
        # So we can maintain adj_count[v] = number of used vertices adjacent to v.
        # Then c is valid iff not used[c] and adj_count[c] == 1.
        
        adj_count = [0] * MAX_VERTS
        # Initialize: v0 is used, so all its neighbors have adj_count incremented
        for nb in neighbors[v0]:
            adj_count[nb] += 1
        
        if not neighbors[v0]:
            return path
            
        v1 = random.choice(neighbors[v0])
        path.append(v1)
        used[v1] = True
        # Update adj_count for v1's neighbors
        for nb in neighbors[v1]:
            adj_count[nb] += 1
        
        def is_valid(candidate, end_vertex):
            if used[candidate]:
                return False
            # candidate is adjacent to end_vertex (by construction)
            # It's valid if adj_count[candidate] == 1 (only adjacent to end_vertex among used)
            return adj_count[candidate] == 1

        def count_onward_moves(candidate):
            """
            Count valid moves from candidate if it were added to the end.
            candidate will be the new end.
            A vertex w is a valid move from candidate if:
            - w is a neighbor of candidate
            - w is not used
            - w is not adjacent to any used vertex except candidate
            After adding candidate, candidate becomes used. So we need to check
            against the current used set PLUS candidate.
            For w to be valid after candidate is added:
            - not used[w]
            - w is adjacent to candidate (true by construction)
            - w is not adjacent to any other used vertex
            Currently, adj_count[w] counts used vertices adjacent to w.
            After adding candidate, if w is adjacent to candidate, adj_count[w] would increase by 1.
            But we need: after adding candidate, the only used vertex adjacent to w is candidate.
            So currently, w should not be adjacent to any used vertex (adj_count[w] == 0),
            OR w should be adjacent to exactly one used vertex which is... wait.
            
            Let me re-derive. Currently used set is U. candidate c is not in U.
            After adding c, used set is U ∪ {c}.
            For w (neighbor of c) to be a valid next move:
            - w not in U ∪ {c} → w not in U and w ≠ c (w ≠ c since w is neighbor)
            - w is not adjacent to any vertex in (U ∪ {c}) except c
            - w is adjacent to c (by construction)
            - So w must not be adjacent to any vertex in U.
            - Currently, adj_count[w] = |{u ∈ U : u adjacent to w}|
            - So we need adj_count[w] == 0.
            
            Wait, but w could be adjacent to some u in U. If so, then after adding c,
            w would be adjacent to both u and c, which violates the induced path condition
            (w and u would be non-consecutive but adjacent).
            
            So: count_onward_moves(c) = |{w ∈ N(c) : not used[w] and adj_count[w] == 0}|
            """
            count = 0
            for w in neighbors[candidate]:
                if used[w]:
                    continue
                if adj_count[w] == 0:
                    count += 1
            return count

        while True:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            valid_left = []
            for v in neighbors[left_end]:
                if is_valid(v, left_end):
                    valid_left.append(v)
            
            # Get valid moves for right end
            valid_right = []
            for v in neighbors[right_end]:
                if is_valid(v, right_end):
                    valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                for v in valid_left:
                    score = count_onward_moves(v)
                    if score < min_score_left:
                        min_score_left = score
                        best_moves_left = [v]
                    elif score == min_score_left:
                        best_moves_left.append(v)
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                for v in valid_right:
                    score = count_onward_moves(v)
                    if score < min_score_right:
                        min_score_right = score
                        best_moves_right = [v]
                    elif score == min_score_right:
                        best_moves_right.append(v)

            if not valid_left:
                next_v = random.choice(best_moves_right)
                path.append(next_v)
                used[next_v] = True
                for nb in neighbors[next_v]:
                    adj_count[nb] += 1
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                path.insert(0, next_v)
                used[next_v] = True
                for nb in neighbors[next_v]:
                    adj_count[nb] += 1
            else:
                # Choose side based on score (tightness) - prefer more constrained
                if min_score_left < min_score_right:
                    next_v = random.choice(best_moves_left)
                    path.insert(0, next_v)
                    used[next_v] = True
                    for nb in neighbors[next_v]:
                        adj_count[nb] += 1
                elif min_score_right < min_score_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used[next_v] = True
                    for nb in neighbors[next_v]:
                        adj_count[nb] += 1
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.insert(0, next_v)
                        used[next_v] = True
                        for nb in neighbors[next_v]:
                            adj_count[nb] += 1
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used[next_v] = True
                        for nb in neighbors[next_v]:
                            adj_count[nb] += 1

        return path

    def polish_path(path, time_budget):
        """
        Try to extend the path by inserting vertices or extending ends.
        Uses local search: try to insert an unused vertex between two consecutive
        path vertices if it maintains the induced path property.
        """
        if not path or len(path) < 2:
            return path
        
        used = [False] * MAX_VERTS
        for v in path:
            used[v] = True
        
        # Precompute adjacency for quick lookup
        # Try extending ends first
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            
            # Try to extend left end
            left_end = path[0]
            for v in neighbors[left_end]:
                if used[v]:
                    continue
                # Check if v is adjacent to any path vertex except left_end
                valid = True
                for i, p in enumerate(path):
                    if i == 0:
                        continue
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.insert(0, v)
                    used[v] = True
                    improved = True
                    break
            
            # Try to extend right end
            if not improved:
                right_end = path[-1]
                for v in neighbors[right_end]:
                    if used[v]:
                        continue
                    valid = True
                    for i, p in enumerate(path):
                        if i == len(path) - 1:
                            continue
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        path.append(v)
                        used[v] = True
                        improved = True
                        break
            
            # Try inserting a vertex between consecutive path vertices
            if not improved:
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i + 1]
                    # Find common neighbors of a and b that are not used
                    # and not adjacent to any other path vertex
                    for v in neighbors[a]:
                        if used[v] or v == b:
                            continue
                        # v must be adjacent to b
                        if (v ^ b).bit_count() != 1:
                            continue
                        # Check v is not adjacent to any other path vertex
                        valid = True
                        for j, p in enumerate(path):
                            if j == i or j == i + 1:
                                continue
                            if (v ^ p).bit_count() == 1:
                                valid = False
                                break
                        if valid:
                            path.insert(i + 1, v)
                            used[v] = True
                            improved = True
                            break
                    if improved:
                        break
        
        return path

    # Multi-start
    # Use random start vertices to break symmetry
    while time.time() - start_time < DEADLINE:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 2.0:
            break
        v0 = random.randint(0, MAX_VERTS - 1)
        # Allocate most time to growth, some to polish
        grow_time = remaining * 0.7
        p = bidirectional_grow(start_vertex=v0, time_limit=grow_time)
        
        # Polish the path
        polish_budget = min(remaining * 0.3, 3.0)
        p = polish_path(p, start_time + DEADLINE - 1.0)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Final polish on best path
    if best_path and (time.time() - start_time) < DEADLINE - 1.0:
        final_budget = DEADLINE - 1.0
        best_path = polish_path(best_path, final_budget)
        best_len = len(best_path)

    if best_path is None:
        # Fallback
        return [0]

    return best_path

# EVOLVE-BLOCK-END
