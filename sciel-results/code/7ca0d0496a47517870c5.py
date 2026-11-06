import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55
    best_len = 0
    best_seq = []
    
    def check_collision(P_new, prefixes, used_sums):
        """Check if adding prefix P_new creates any sum collision."""
        for i in range(len(prefixes)):
            d = P_new - prefixes[i]
            if d in used_sums:
                return False, None
        return True, None
    
    def backtrack(seq, used_sums, prefixes):
        nonlocal best_len, best_seq
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Time guard
        if time.time() > deadline:
            return
            
        last_x = seq[-1] if seq else 0
        P_last = prefixes[-1]
        
        # Heuristic: try candidates with moderate gaps first to spread sums
        # This helps avoid collisions
        candidates = list(range(last_x + 1, 257))
        # Sort by a heuristic: prefer gaps that are neither too small nor too large
        # A gap around 10-30 tends to work well
        def gap_heuristic(val):
            gap = val - last_x
            # Prefer gaps in range [8, 40]
            if 8 <= gap <= 40:
                return 0
            elif 4 <= gap < 8 or 40 < gap <= 60:
                return 1
            else:
                return 2
        candidates.sort(key=lambda v: (gap_heuristic(v), v))
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            P_new = P_last + x
            
            # Check for collisions with existing sums
            valid = True
            new_diffs = []
            for p in prefixes:
                d = P_new - p
                if d in used_sums:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Periodic time check
            if time.time() > deadline:
                return
                
            # Forward checking: verify that at least one valid extension exists
            can_extend = False
            for y in range(x + 1, 257):
                P_next = P_new + y
                extend_valid = True
                # Check against all existing prefixes (including P_new)
                for p in prefixes:
                    d = P_next - p
                    if d in used_sums:
                        extend_valid = False
                        break
                if extend_valid:
                    # Check against the new differences (P_new - p for p in prefixes)
                    for p in prefixes:
                        d = P_next - p
                        # d should not equal any existing sum, already checked
                        # But also need to ensure d is not in new_diffs
                        # new_diffs contains P_new - p_old for each p_old in prefixes
                        # So we need: P_next - p_old != P_new - p_old' for all pairs
                        # That is: y != P_new - p_old' + p_old - P_new... 
                        # Actually simpler: the new sum from P_next to p_old is P_next - p_old
                        # This must not be in used_sums (checked) and must not equal
                        # any of the new_diffs which are P_new - p_old' for p_old' in prefixes
                        # So check: P_next - p_old not in new_diffs
                        if d in new_diffs:
                            extend_valid = False
                            break
                if extend_valid:
                    can_extend = True
                    break
                
            if not can_extend and len(seq) + 1 <= best_len:
                continue
                
            # Commit
            seq.append(x)
            prefixes.append(P_new)
            used_sums.update(new_diffs)
            
            # Recurse
            backtrack(seq, used_sums, prefixes)
            
            # Undo
            seq.pop()
            prefixes.pop()
            for d in new_diffs:
                used_sums.discard(d)
                
    backtrack([], set(), [0])
    return best_seq
