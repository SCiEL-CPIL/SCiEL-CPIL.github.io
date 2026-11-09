"""Bidirectional Warnsdorff + CSP repair snake construction for Q_9."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START

def _deadline_hit(start_time, margin=12.0):
    return time.time() - start_time > 120.0 - margin

def _is_valid_extension(v, path, used_set):
    if v in used_set:
        return False
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _is_valid_extension_head(v, path, used_set):
    if v in used_set:
        return False
    for p in path[1:]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _bidirectional_grow(path, used_set, rng, start_time, max_iters=None):
    """Bidirectional growth with Warnsdorff's rule."""
    if max_iters is None:
        max_iters = 1000
    for _ in range(max_iters):
        if _deadline_hit(start_time):
            break
        tail_options = []
        for b in range(N):
            v = path[-1] ^ (1 << b)
            if _is_valid_extension(v, path, used_set):
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used_set:
                        continue
                    valid = True
                    for p in path[:-1]:
                        if (v2 ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        sub += 1
                tail_options.append((sub, b, v))
        head_options = []
        for b in range(N):
            v = path[0] ^ (1 << b)
            if _is_valid_extension_head(v, path, used_set):
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used_set:
                        continue
                    valid = True
                    for p in path[1:]:
                        if (v2 ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        sub += 1
                head_options.append((sub, b, v))
        if not tail_options and not head_options:
            break
        if not tail_options:
            choose_head = True
        elif not head_options:
            choose_head = False
        else:
            tail_min = min(o[0] for o in tail_options)
            head_min = min(o[0] for o in head_options)
            if head_min < tail_min:
                choose_head = True
            elif tail_min < head_min:
                choose_head = False
            else:
                choose_head = rng.random() < 0.5
        if choose_head:
            best = min(head_options, key=lambda x: x[0])
            candidates = [o for o in head_options if o[0] == best[0]]
            chosen = rng.choice(candidates)
            v = chosen[2]
            path.insert(0, v)
            used_set.add(v)
        else:
            best = min(tail_options, key=lambda x: x[0])
            candidates = [o for o in tail_options if o[0] == best[0]]
            chosen = rng.choice(candidates)
            v = chosen[2]
            path.append(v)
            used_set.add(v)
    return path

def _validate(path):
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _build_forbidden_mask(path):
    """
    Build a bitmask where bit i is set if vertex i is 'forbidden'
    (adjacent to some path vertex at distance >= 2 from the insertion point).
    For tail-extension: forbidden[i] = 1 if i is adjacent to any path[0:-1] vertex.
    For head-extension: forbidden[i] = 1 if i is adjacent to any path[1:] vertex.
    """
    mask = 0
    for p in path:
        for b in range(N):
            mask |= 1 << (p ^ (1 << b))
    return mask

def _csp_extend_tail(path, used_set, start_time, max_steps=200):
    """
    CSP-style tail extension using a precomputed forbidden mask.
    At each step, find all neighbors of the current tail that are not used
    and not forbidden (adjacent to any non-last path vertex).
    Use Warnsdorff ordering among valid candidates.
    """
    current = path[:]
    used = set(used_set)
    # Build forbidden mask: vertices adjacent to path[0:-1] (all except last)
    # A candidate v at tail is valid iff v not in used AND v not adjacent to any path[0:-1]
    # We maintain: forbidden_mask has bits set for all neighbors of path[0:-1]
    forbidden_mask = 0
    for p in current[:-1]:
        for b in range(N):
            forbidden_mask |= 1 << (p ^ (1 << b))
    
    for _ in range(max_steps):
        if _deadline_hit(start_time):
            break
        cur = current[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if (forbidden_mask >> v) & 1:
                continue
            # v is valid: not used, not adjacent to any path[0:-1]
            # Count its own forbidden neighbors (for Warnsdorff)
            sub = 0
            for b2 in range(N):
                v2 = v ^ (1 << b2)
                if v2 in used:
                    continue
                if (forbidden_mask >> v2) & 1:
                    continue
                if v2 == cur:
                    continue
                sub += 1
            candidates.append((sub, v))
        
        if not candidates:
            break
        
        # Warnsdorff: pick candidate with fewest subsequent options
        candidates.sort(key=lambda x: x[0])
        best_sub = candidates[0][0]
        ties = [v for s, v in candidates if s == best_sub]
        v = random.choice(ties)
        
        current.append(v)
        used.add(v)
        # Update forbidden mask: v's neighbors become forbidden for future
        for b in range(N):
            forbidden_mask |= 1 << (v ^ (1 << b))
    
    return current

def _csp_extend_head(path, used_set, start_time, max_steps=200):
    """CSP-style head extension using forbidden mask."""
    current = path[:]
    used = set(used_set)
    forbidden_mask = 0
    for p in current[1:]:
        for b in range(N):
            forbidden_mask |= 1 << (p ^ (1 << b))
    
    for _ in range(max_steps):
        if _deadline_hit(start_time):
            break
        cur = current[0]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if (forbidden_mask >> v) & 1:
                continue
            sub = 0
            for b2 in range(N):
                v2 = v ^ (1 << b2)
                if v2 in used:
                    continue
                if (forbidden_mask >> v2) & 1:
                    continue
                if v2 == cur:
                    continue
                sub += 1
            candidates.append((sub, v))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        best_sub = candidates[0][0]
        ties = [v for s, v in candidates if s == best_sub]
        v = random.choice(ties)
        
        current.insert(0, v)
        used.add(v)
        for b in range(N):
            forbidden_mask |= 1 << (v ^ (1 << b))
    
    return current

def _csp_bidirectional(path, used_set, rng, start_time, max_iters=500):
    """
    Bidirectional CSP extension: alternate between extending head and tail
    using the fast forbidden-mask check. Choose the more constrained end.
    """
    current = path[:]
    used = set(used_set)
    
    # Build both forbidden masks
    tail_forbidden = 0  # neighbors of current[:-1]
    head_forbidden = 0  # neighbors of current[1:]
    for p in current[:-1]:
        for b in range(N):
            tail_forbidden |= 1 << (p ^ (1 << b))
    for p in current[1:]:
        for b in range(N):
            head_forbidden |= 1 << (p ^ (1 << b))
    
    for _ in range(max_iters):
        if _deadline_hit(start_time):
            break
        
        # Count valid tail candidates
        cur_tail = current[-1]
        tail_cands = []
        for b in range(N):
            v = cur_tail ^ (1 << b)
            if v in used or (tail_forbidden >> v) & 1:
                continue
            tail_cands.append(v)
        
        # Count valid head candidates
        cur_head = current[0]
        head_cands = []
        for b in range(N):
            v = cur_head ^ (1 << b)
            if v in used or (head_forbidden >> v) & 1:
                continue
            head_cands.append(v)
        
        if not tail_cands and not head_cands:
            break
        
        # Choose more constrained end
        if not tail_cands:
            choose_head = True
        elif not head_cands:
            choose_head = False
        else:
            choose_head = rng.random() < 0.5
        
        if choose_head:
            # Pick from head_cands with Warnsdorff
            scored = []
            for v in head_cands:
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used or (head_forbidden >> v2) & 1 or v2 == cur_head:
                        continue
                    sub += 1
                scored.append((sub, v))
            scored.sort(key=lambda x: x[0])
            best_sub = scored[0][0]
            ties = [v for s, v in scored if s == best_sub]
            v = rng.choice(ties)
            current.insert(0, v)
            used.add(v)
            for b in range(N):
                head_forbidden |= 1 << (v ^ (1 << b))
                tail_forbidden |= 1 << (v ^ (1 << b))
        else:
            scored = []
            for v in tail_cands:
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used or (tail_forbidden >> v2) & 1 or v2 == cur_tail:
                        continue
                    sub += 1
                scored.append((sub, v))
            scored.sort(key=lambda x: x[0])
            best_sub = scored[0][0]
            ties = [v for s, v in scored if s == best_sub]
            v = rng.choice(ties)
            current.append(v)
            used.add(v)
            for b in range(N):
                head_forbidden |= 1 << (v ^ (1 << b))
                tail_forbidden |= 1 << (v ^ (1 << b))
    
    return current

def _backtrack_repair(path, start_time, max_depth=5, max_branch=3):
    """
    Backtracking repair: if the path is stuck, try removing the last k vertices
    and exploring alternative branches. This is a limited-depth CSP backtracking
    that uses the forbidden mask for fast constraint checking.
    """
    if len(path) < 3:
        return path
    
    best = path[:]
    
    # Try removing last 1 to max_depth vertices and re-exploring
    for depth in range(1, min(max_depth + 1, len(path) - 1)):
        if _deadline_hit(start_time):
            break
        
        base = path[:-depth]
        used = set(base)
        
        # Build forbidden mask for base[:-1] (tail extension)
        forbidden = 0
        for p in base[:-1]:
            for b in range(N):
                forbidden |= 1 << (p ^ (1 << b))
        
        # BFS/DFS with limited branching
        stack = [(base, used, forbidden)]
        found = False
        for _ in range(max_branch * depth * 10):
            if _deadline_hit(start_time):
                break
            if not stack:
                break
            cur_path, cur_used, cur_forbidden = stack.pop()
            cur = cur_path[-1]
            
            cands = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in cur_used or (cur_forbidden >> v) & 1:
                    continue
                cands.append(v)
            
            if not cands:
                continue
            
            # Sort by Warnsdorff
            cands.sort(key=lambda v: sum(
                1 for b2 in range(N)
                if (v ^ (1 << b2)) not in cur_used
                and not ((cur_forbidden >> (v ^ (1 << b2))) & 1)
                and (v ^ (1 << b2)) != cur
            ))
            
            for v in cands[:max_branch]:
                new_path = cur_path + [v]
                new_used = cur_used | {v}
                new_forbidden = cur_forbidden
                for b in range(N):
                    new_forbidden |= 1 << (v ^ (1 << b))
                
                if len(new_path) > len(best):
                    best = new_path[:]
                
                # Push to stack for further exploration (limited)
                if len(new_path) < len(path) + 20:
                    stack.append((new_path, new_used, new_forbidden))
        
        if len(best) >= len(path):
            break
    
    return best

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Combines bidirectional Warnsdorff growth with CSP forbidden-mask extension
    and limited backtracking repair for global structural improvement.
    """
    start_time = time.time()
    
    best_path = None
    best_len = 0
    
    def _try_bidirectional(start_v, rng):
        path = [start_v]
        used = {start_v}
        # Seed with 3 forward steps
        for _ in range(3):
            cands = []
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    cands.append(v)
            if not cands:
                break
            v = rng.choice(cands)
            path.append(v)
            used.add(v)
        path = _bidirectional_grow(path, used, rng, start_time)
        return path
    
    def _try_csp(start_v, rng):
        path = [start_v]
        used = {start_v}
        for _ in range(2):
            cands = []
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    cands.append(v)
            if not cands:
                break
            v = rng.choice(cands)
            path.append(v)
            used.add(v)
        path = _csp_bidirectional(path, used, rng, start_time, max_iters=800)
        return path
    
    def _local_extend(path, rng):
        """Fast local extension using forbidden masks."""
        current = path[:]
        used = set(current)
        for _ in range(100):
            if _deadline_hit(start_time):
                break
            # Try tail
            cur = current[-1]
            cands = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and _is_valid_extension(v, current, used):
                    cands.append(v)
            if cands:
                v = rng.choice(cands)
                current.append(v)
                used.add(v)
                continue
            # Try head
            cur = current[0]
            cands = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and _is_valid_extension_head(v, current, used):
                    cands.append(v)
            if cands:
                v = rng.choice(cands)
                current.insert(0, v)
                used.add(v)
                continue
            break
        return current
    
    # Phase 1: Bidirectional Warnsdorff restarts
    for attempt in range(150):
        if _deadline_hit(start_time):
            break
        rng = random.Random(attempt * 7919 + 13)
        path = _try_bidirectional(0, rng)
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
    
    # Phase 2: CSP-based restarts
    for attempt in range(100):
        if _deadline_hit(start_time):
            break
        rng = random.Random(attempt * 104729 + 42)
        sv = rng.randint(0, MAX_VERTS - 1)
        path = _try_csp(sv, rng)
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
    
    # Phase 3: Backtracking repair on best
    if best_path and not _deadline_hit(start_time):
        repaired = _backtrack_repair(best_path, start_time, max_depth=4, max_branch=4)
        if len(repaired) > best_len:
            best_path = repaired[:]
            best_len = len(best_path)
    
    # Phase 4: Local extension on best
    if best_path and not _deadline_hit(start_time):
        extended = _local_extend(best_path, random.Random(77777))
        if len(extended) > best_len:
            best_path = extended[:]
            best_len = len(best_path)
    
    # Phase 5: More mixed restarts if time remains
    for attempt in range(300):
        if _deadline_hit(start_time):
            break
        rng = random.Random(attempt * 31337 + 99)
        sv = rng.randint(0, MAX_VERTS - 1)
        if attempt % 3 == 0:
            path = _try_bidirectional(sv, rng)
        else:
            path = _try_csp(sv, rng)
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        # Periodic repair
        if attempt % 20 == 19 and best_path and not _deadline_hit(start_time):
            repaired = _backtrack_repair(best_path, start_time, max_depth=3, max_branch=3)
            if len(repaired) > best_len:
                best_path = repaired[:]
                best_len = len(best_path)
    
    # Phase 6: Final local extension
    if best_path and not _deadline_hit(start_time):
        extended = _local_extend(best_path, random.Random(424242))
        if len(extended) > best_len:
            best_path = extended[:]
            best_len = len(best_path)
    
    if best_path is None:
        path, used = [0], {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path
    
    return best_path
# EVOLVE-BLOCK-END
