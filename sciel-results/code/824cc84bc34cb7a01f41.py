import time
import random

N = 9
V = 2 ** N

# Precompute neighbors for all vertices
neighbors = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    best_path = []
    
    # Try multiple starting vertices
    # We can shuffle the vertices or just iterate
    # 512 vertices is small.
    
    for start in range(V):
        if time.time() - start_time > timeout:
            break
            
        path = [start]
        used = {start}
        # blocked_all: vertices w such that dist(w, p) <= 1 for some p in path
        blocked_all = set()
        blocked_all.add(start)
        blocked_all.update(neighbors[start])
        
        current_len = 1
        
        while True:
            if time.time() - start_time > timeout:
                break
                
            left = path[0]
            right = path[-1]
            
            # Find valid candidates for left
            valid_left = []
            for u in neighbors[left]:
                if u in used:
                    continue
                # Check dist >= 2 from all p in path[1:]
                ok = True
                for p in path[1:]:
                    if (u ^ p).bit_count() < 2:
                        ok = False
                        break
                if ok:
                    valid_left.append(u)
                    
            # Find valid candidates for right
            valid_right = []
            for u in neighbors[right]:
                if u in used:
                    continue
                # Check dist >= 2 from all p in path[:-1]
                ok = True
                for p in path[:-1]:
                    if (u ^ p).bit_count() < 2:
                        ok = False
                        break
                if ok:
                    valid_right.append(u)
            
            if not valid_left and not valid_right:
                break
            
            # Decide which end to extend
            # Heuristic: extend the end with fewer options
            if len(valid_left) < len(valid_right):
                candidates = valid_left
                extend_left = True
            elif len(valid_right) < len(valid_left):
                candidates = valid_right
                extend_left = False
            else:
                # Tie: pick randomly or alternate? 
                # Random adds diversity
                if random.random() < 0.5:
                    candidates = valid_left
                    extend_left = True
                else:
                    candidates = valid_right
                    extend_left = False
            
            if not candidates:
                break
                
            # Apply Warnsdorff: pick candidate with fewest future options
            # Future options for u: count of w in neighbors[u] such that w not in blocked_all
            # Note: blocked_all is updated AFTER we add u. 
            # But the "future options" for u (as the new end) depend on the path INCLUDING u.
            # So we should use the updated blocked_all?
            # No, the options for the NEXT step from u are neighbors w of u.
            # w is valid if dist(w, p) >= 2 for all p in path (old path) and w not in used.
            # This is exactly w not in blocked_all (old) and w != u?
            # If w is in blocked_all (old), it means dist(w, p) <= 1 for some p in old path.
            # So w is invalid.
            # So score(u) = count of w in neighbors[u] if w not in blocked_all.
            
            best_u = None
            best_score = float('inf')
            
            for u in candidates:
                score = 0
                for w in neighbors[u]:
                    if w not in blocked_all:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_u = u
            
            if best_u is None:
                break
                
            # Update path
            if extend_left:
                path.insert(0, best_u)
            else:
                path.append(best_u)
                
            used.add(best_u)
            blocked_all.add(best_u)
            blocked_all.update(neighbors[best_u])
            
            current_len += 1
            
            if len(path) > len(best_path):
                best_path = path[:]
                
        # End while for this start
    
    return best_path
