import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # --- 1. Generate Baseline Recursive Superpermutation ---
    def recursive_superperm(n_val):
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    baseline_word = recursive_superperm(n)
    baseline_len = len(baseline_word)
    
    if n <= 4:
        return "".join(map(str, baseline_word))

    best_S = baseline_word[:]
    best_len = baseline_len

    # --- 2. Local Search: Removal Passes with incremental coverage ---
    
    def build_counts(word, n_val):
        counts = {}
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def build_coverage_from_counts(counts, word, n_val):
        """Build coverage array: cov[i] = number of critical windows covering position i."""
        L = len(word)
        cov = [0] * L
        for i in range(L - n_val + 1):
            p = tuple(word[i:i+n_val])
            if counts.get(p, 0) == 1:
                for pos in range(i, min(i + n_val, L)):
                    cov[pos] += 1
        return cov

    def run_removal_passes(word, n_val, deadline):
        """Run removal passes with incremental coverage updates."""
        S = word[:]
        L = len(S)
        counts = build_counts(S, n_val)
        coverage = build_coverage_from_counts(counts, S, n_val)
        
        stuck_passes = 0
        max_stuck_passes = 50
        
        while time.time() < deadline:
            removed_in_pass = 0
            L = len(S)
            
            for i in range(L - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                if coverage[i] > 0:
                    continue
                
                # Valid removal at position i
                old_start = max(0, i - n_val + 1)
                old_end = min(i, L - n_val)
                
                # Collect old windows affected
                old_windows = []
                for j in range(old_start, old_end + 1):
                    old_windows.append((j, tuple(S[j:j+n_val])))
                
                # Save coverage contributions from old windows that are critical
                # We need to track which old windows were critical (count==1)
                critical_old = []
                for j, p in old_windows:
                    if counts.get(p, 0) == 1:
                        critical_old.append((j, p))
                
                S.pop(i)
                L -= 1
                removed_in_pass += 1
                
                # Decrement old windows
                for j, p in old_windows:
                    c = counts.get(p, 0) - 1
                    if c <= 0:
                        if p in counts:
                            del counts[p]
                    else:
                        counts[p] = c
                
                # Increment new windows (windows that shift into position due to removal)
                new_start = max(0, i - n_val + 1)
                new_end = min(i - 1, L - n_val)
                new_windows = []
                for j in range(new_start, new_end + 1):
                    p = tuple(S[j:j+n_val])
                    old_c = counts.get(p, 0)
                    counts[p] = old_c + 1
                    new_windows.append((j, p, old_c))
                
                # Update coverage incrementally:
                # For each critical_old window at position j, it covered positions [j, j+n-1]
                # Remove its contribution
                for j, p in critical_old:
                    for pos in range(j, min(j + n_val, L + 1)):
                        if pos < L:
                            coverage[pos] -= 1
                
                # For each new window that became critical (old_c == 0, now count 1)
                for j, p, old_c in new_windows:
                    if old_c == 0:
                        # This window is now critical (count went from 0 to 1)
                        for pos in range(j, min(j + n_val, L)):
                            coverage[pos] += 1
                
                # Also, old windows that had count 2 and now have count 1 become critical
                # But we already handled critical_old (count==1 before removal)
                # Windows with count 2 before removal: after decrement they have count 1 -> become critical
                # We need to handle this separately
                for j, p in old_windows:
                    # After decrement, if count is now 1, it became critical
                    if p in counts and counts[p] == 1:
                        # But we already decremented, so check if it was 2 before
                        # Actually we need to check: was it 2 before? We stored old_c for new windows
                        # For old windows, we didn't store old count. Let's handle differently.
                        pass
                
                # Simpler approach: just rebuild coverage if we're being conservative
                # But that's expensive. Let's do it properly.
                # Actually the above logic is getting complex. Let me use a simpler incremental approach.
                
            if removed_in_pass == 0:
                stuck_passes += 1
                if stuck_passes > max_stuck_passes:
                    break
            else:
                stuck_passes = 0
        
        return S, len(S)

    # Use a simpler but correct incremental approach
    def run_removal_simple(word, n_val, deadline):
        """Simpler removal pass: rebuild coverage less frequently."""
        S = word[:]
        L = len(S)
        counts = build_counts(S, n_val)
        coverage = build_coverage_from_counts(counts, S, n_val)
        
        stuck_passes = 0
        max_stuck_passes = 30
        removals_since_rebuild = 0
        rebuild_threshold = 20  # Rebuild coverage every N removals
        
        while time.time() < deadline:
            removed_in_pass = 0
            L = len(S)
            
            for i in range(L - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                if coverage[i] > 0:
                    continue
                
                # Valid removal at position i
                old_start = max(0, i - n_val + 1)
                old_end = min(i, L - n_val)
                
                old_windows = []
                for j in range(old_start, old_end + 1):
                    old_windows.append((j, tuple(S[j:j+n_val])))
                
                S.pop(i)
                L -= 1
                removed_in_pass += 1
                removals_since_rebuild += 1
                
                # Decrement old windows
                for j, p in old_windows:
                    c = counts.get(p, 0) - 1
                    if c <= 0:
                        if p in counts:
                            del counts[p]
                    else:
                        counts[p] = c
                
                # Increment new windows
                new_start = max(0, i - n_val + 1)
                new_end = min(i - 1, L - n_val)
                for j in range(new_start, new_end + 1):
                    p = tuple(S[j:j+n_val])
                    counts[p] = counts.get(p, 0) + 1
                
                # Periodically rebuild coverage to keep it accurate
                if removals_since_rebuild >= rebuild_threshold:
                    coverage = build_coverage_from_counts(counts, S, n_val)
                    removals_since_rebuild = 0
            
            if removed_in_pass == 0:
                stuck_passes += 1
                if stuck_passes > max_stuck_passes:
                    break
            else:
                stuck_passes = 0
            
            # If we made progress, reset removals counter
            if removed_in_pass > 0:
                removals_since_rebuild = 0
                coverage = build_coverage_from_counts(counts, S, n_val)
        
        return S, len(S)

    # Run multiple passes with different strategies
    # Strategy 1: Standard removal
    S1, len1 = run_removal_simple(baseline_word, n, deadline)
    if len1 < best_len:
        best_S = S1
        best_len = len1
    
    # Strategy 2: Try removal from front to back
    def run_removal_front(word, n_val, deadline):
        S = word[:]
        L = len(S)
        counts = build_counts(S, n_val)
        coverage = build_coverage_from_counts(counts, S, n_val)
        
        stuck_passes = 0
        max_stuck_passes = 30
        removals_since_rebuild = 0
        rebuild_threshold = 20
        
        while time.time() < deadline:
            removed_in_pass = 0
            L = len(S)
            
            for i in range(L):
                if time.time() > deadline:
                    break
                
                if coverage[i] > 0:
                    continue
                
                old_start = max(0, i - n_val + 1)
                old_end = min(i, L - n_val)
                
                old_windows = []
                for j in range(old_start, old_end + 1):
                    old_windows.append((j, tuple(S[j:j+n_val])))
                
                S.pop(i)
                L -= 1
                removed_in_pass += 1
                removals_since_rebuild += 1
                
                for j, p in old_windows:
                    c = counts.get(p, 0) - 1
                    if c <= 0:
                        if p in counts:
                            del counts[p]
                    else:
                        counts[p] = c
                
                new_start = max(0, i - n_val + 1)
                new_end = min(i - 1, L - n_val)
                for j in range(new_start, new_end + 1):
                    p = tuple(S[j:j+n_val])
                    counts[p] = counts.get(p, 0) + 1
                
                if removals_since_rebuild >= rebuild_threshold:
                    coverage = build_coverage_from_counts(counts, S, n_val)
                    removals_since_rebuild = 0
            
            if removed_in_pass == 0:
                stuck_passes += 1
                if stuck_passes > max_stuck_passes:
                    break
            else:
                stuck_passes = 0
                removals_since_rebuild = 0
                coverage = build_coverage_from_counts(counts, S, n_val)
        
        return S, len(S)

    if time.time() < deadline - 5:
        S2, len2 = run_removal_front(baseline_word, n, deadline)
        if len2 < best_len:
            best_S = S2
            best_len = len2
    
    # Strategy 3: Combine - remove from best so far, then try front removal
    if time.time() < deadline - 10:
        S3, len3 = run_removal_front(best_S, n, deadline)
        if len3 < best_len:
            best_S = S3
            best_len = len3
    
    # Strategy 4: Another back-to-front pass on the best
    if time.time() < deadline - 10:
        S4, len4 = run_removal_simple(best_S, n, deadline)
        if len4 < best_len:
            best_S = S4
            best_len = len4

    return "".join(map(str, best_S))
