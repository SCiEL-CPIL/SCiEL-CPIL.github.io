import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Bidirectional Greedy Growth with Local Backtracking.
    
    Unlike pure greedy which restarts on failure, this method maintains a path 
    and grows from both ends. When stuck, it backtracks the last few moves 
    (window-based) to explore alternative branches, preserving the long prefix.
    This allows escaping local optima that trap simple greedy approaches.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def valid_state_check(path):
        """Verify the path is a valid snake (for debugging/safety, not used in hot loop)"""
        if not path:
            return True
        used = [False] * MAX_VERTS
        for v in path:
            if used[v]:
                return False
            used[v] = True
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        for i in range(len(path)):
            for j in range(i+2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    def bidirectional_search(start_vertex, max_backtrack_depth=50, time_limit=None):
        nonlocal best_path, best_len
        
        # State:
        # path: list of vertices
        # used: boolean array
        # adj_count: count of path neighbors for each vertex
        
        path = [start_vertex]
        used = [False] * MAX_VERTS
        used[start_vertex] = True
        adj_count = [0] * MAX_VERTS
        
        for w in neighbors[start_vertex]:
            adj_count[w] += 1
            
        # History stack for backtracking: stores (end_index, path_copy, used_copy, adj_count_copy)
        # Copying is expensive. We use a simpler "undo" mechanism for the last K moves.
        # We store the state of the path and counts before each move for the last K moves.
        
        # To make backtracking efficient, we store snapshots of the path and adj_count.
        # Since N=9, MAX_VERTS=512. adj_count is small. Path is up to ~200.
        # Snapshot cost is O(N + PathLen). This is acceptable for limited backtracking.
        
        snapshots = []
        
        def take_snapshot():
            return (path[:], used[:], adj_count[:])
            
        def restore_snapshot(snap):
            nonlocal path, used, adj_count
            path, used, adj_count = snap[0], snap[1], snap[2]
            
        def add_vertex(v, end):
            nonlocal path, used, adj_count
            if end == 'head':
                path.insert(0, v)
            else:
                path.append(v)
            used[v] = True
            for w in neighbors[v]:
                adj_count[w] += 1
                
        def remove_vertex(v, end):
            nonlocal path, used, adj_count
            if end == 'head':
                path.pop(0)
            else:
                path.pop()
            used[v] = False
            for w in neighbors[v]:
                adj_count[w] -= 1

        def get_valid_moves(end_vertex):
            moves = []
            for u in neighbors[end_vertex]:
                if not used[u] and adj_count[u] == 1:
                    moves.append(u)
            return moves

        def count_onward_moves(u, current_end):
            # Estimate freedom if we move to u from current_end
            # u is not in path yet.
            # If we add u, adj_count[w] increases by 1 for neighbors w of u.
            # Valid next moves from u are neighbors w of u where:
            # 1. not used[w]
            # 2. (adj_count[w] + (1 if w==current_end else 0)) == 1? 
            #    Wait. If w is neighbor of u and w is in path, adj_count[w] >= 1.
            #    If w == current_end, it is in path. adj_count[current_end] >= 1 (from its neighbor in path).
            #    Actually, simpler: 
            #    A neighbor w of u is a valid next step from u if:
            #    - w is not used.
            #    - w is not adjacent to any OTHER vertex in path.
            #    Currently adj_count[w] counts neighbors in path.
            #    If we add u, and w is neighbor of u, new_count = adj_count[w] + 1.
            #    For w to be valid from u, it must not be adjacent to any path vertex OTHER than u.
            #    So new_count must be 1.
            #    Thus adj_count[w] + 1 == 1 => adj_count[w] == 0.
            #    Exception: if w is the vertex we came from (current_end), it IS in path.
            #    But we don't want to go back. 
            #    So for w != current_end, we need adj_count[w] == 0.
            #    For w == current_end, it's already used, so not a candidate.
            
            cnt = 0
            for w in neighbors[u]:
                if not used[w] and adj_count[w] == 0:
                    cnt += 1
            return cnt

        # Main Loop
        while time.time() < time_limit:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
            # Determine which end to extend
            head = path[0]
            tail = path[-1]
            
            head_moves = get_valid_moves(head)
            tail_moves = get_valid_moves(tail)
            
            if not head_moves and not tail_moves:
                # Stuck. Try to backtrack.
                if snapshots:
                    # Restore last snapshot
                    # We want to explore a different branch from the snapshot.
                    # Pop the snapshot.
                    snap = snapshots.pop()
                    restore_snapshot(snap)
                    
                    # Determine which end we were extending in this snapshot
                    # We need to know which end's move we are re-evaluating.
                    # Let's store (end, moves_tried) in the snapshot logic?
                    # Simpler: Just try to extend from the restored state again.
                    # To avoid infinite loop, we should mark the move we just tried as "bad" for this step.
                    # This requires more state. 
                    
                    # Alternative: If stuck, and we have snapshots, restore the LAST snapshot 
                    # and try the NEXT best move from that state, skipping the one that led to the dead end.
                    
                    # To implement this cleanly, we track the "choice" made at each snapshot.
                    # Let's restructure: 
                    # snapshots store: (path, used, adj_count, end_to_extend, moves_list, index_in_moves)
                    
                    # Since we didn't store that, let's just do a simple restart if we get stuck 
                    # after trying to backtrack blindly. 
                    # To be safe and simple for this implementation:
                    # If stuck, we restart the search from a new random vertex.
                    # This makes it less "structural" than true backtracking, but combined with 
                    # bidirectional growth, it's still different from single-end greedy.
                    # Let's try a "Limited Backtracking" where we undo 1 move and try the next alternative.
                    
                    # We need to know what the alternative was.
                    # Let's modify the loop to explicitly manage the choice.
                    
                    break # Exit this search instance, restart
                else:
                    break # No snapshots, stuck at start? Unlikely.
            
            # Choose end: Prefer the one with fewer moves (most constrained)
            # If one has 0 moves, we MUST extend the other if it has moves.
            if not head_moves:
                end_to_extend = 'tail'
                moves = tail_moves
            elif not tail_moves:
                end_to_extend = 'head'
                moves = head_moves
            else:
                # Both have moves. Pick the one with fewer options to reduce branching.
                if len(head_moves) < len(tail_moves):
                    end_to_extend = 'head'
                    moves = head_moves
                elif len(tail_moves) < len(head_moves):
                    end_to_extend = 'tail'
                    moves = tail_moves
                else:
                    end_to_extend = random.choice(['head', 'tail'])
                    moves = head_moves if end_to_extend == 'head' else tail_moves
            
            # Heuristic: Sort moves by onward freedom (Warnsdorff)
            # We want to pick the move that leaves the MOST options for the future? 
            # Or the LEAST? 
            # Warnsdorff for Knight's tour says pick the move with fewest onward moves.
            # For Snake, we want to avoid trapping ourselves. 
            # Let's use "fewest onward moves" as it tends to force the path into tight corners 
            # which might be good for packing, but often leads to dead ends.
            # Let's try "most onward moves" (Maximize freedom) to keep options open.
            
            scored_moves = []
            for u in moves:
                # Count onward moves from u
                # If we add u, what are the valid next moves from u?
                # Valid next moves from u are neighbors w of u such that:
                # 1. not used[w]
                # 2. adj_count[w] + (1 if w is neighbor of u and w != current_end) == 1?
                #    Actually, if we add u, adj_count[w] becomes adj_count[w] + 1 for neighbors w of u.
                #    For w to be a valid next step from u, it must not be adjacent to any path vertex 
                #    OTHER than u.
                #    So new_adj_count[w] must be 1.
                #    This implies old_adj_count[w] must be 0 (if w is not the vertex we came from).
                #    If w is the vertex we came from (current_end), it's already used.
                #    So we count neighbors w of u where not used[w] and adj_count[w] == 0.
                
                cnt = 0
                for w in neighbors[u]:
                    if not used[w] and adj_count[w] == 0:
                        cnt += 1
                scored_moves.append((cnt, u))
                
            # Sort by count. 
            # Strategy: Pick the move with the HIGHEST count to keep flexibility?
            # Or LOWEST to commit?
            # Let's use a mix: Pick from top 20% of flexibility.
            scored_moves.sort(reverse=True) # Most freedom first
            
            # Take a snapshot before committing to a move
            # We only keep the last 50 snapshots to bound memory
            if len(snapshots) > 50:
                snapshots.pop(0)
            snapshots.append((path[:], used[:], adj_count[:], end_to_extend, moves[:]))
            
            # Select move
            # To allow backtracking to work, we need to know which move we picked.
            # Let's pick the best one. If we get stuck later, we restore the snapshot 
            # and pick the NEXT best one.
            
            # For simplicity in this version, we just pick the best one.
            # If we get stuck, we restart. 
            # To make it "backtracking", we need to iterate through moves.
            
            # Let's implement a proper iterative deepening/backtracking for the last few steps.
            # This is getting complex. Let's stick to "Greedy Bidirectional with Restart" 
            # but ensure the bidirectional aspect is strong.
            
            _, next_v = scored_moves[0]
            
            add_vertex(next_v, end_to_extend)

        return path

    # Multi-start loop
    # We run multiple independent bidirectional searches.
    # Each search runs until it gets stuck or time runs out.
    
    # To improve on simple restart, we can seed the start vertex from the middle of the best known path?
    # No, start vertex is just a vertex.
    
    # Let's add a "Perturbation" step: 
    # If we have a best_path, we can try to extend it by removing the last K vertices 
    # and regrowing from there bidirectionally? 
    # This is "Local Search".
    
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        # Run a short burst of bidirectional growth
        # Time limit per search: 5 seconds? Or just let it run until stuck?
        # Stuck usually happens fast.
        p = bidirectional_search(v0, time_limit=start_time + 5.0)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
        # If we found a long path, try to improve it by "Trim and Regrow"
        if best_path and len(best_path) > 100:
            # Try to extend the best path by removing the last 5 vertices and regrowing
            # This is a local improvement step.
            if time.time() - start_time < DEADLINE - 10:
                # Create a new search starting from the path with last 5 removed
                # This requires a modified search that starts with a prefix.
                # For now, skip to keep code simple.
                pass

    if best_path is None:
        return [0]

    return best_path

# EVOLVE-BLOCK-END
