"""Shortest superpermutation via greedy maximal-overlap path with dead-end lookahead."""
import time
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    
    deadline = time.time() + 140.0  # safety margin under 150s kill
    
    def _classic(n):
        """Classic recursive construction: length = 1! + 2! + ... + n!"""
        word = [1]
        for m in range(2, n + 1):
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    def _greedy_superperm(n):
        """Greedy path covering all permutations with dead-end avoidance."""
        all_perms = list(permutations(range(1, n + 1)))
        perm_set = set(all_perms)
        total = len(all_perms)
        
        # For each (n-1)-tuple prefix, which symbols extend it to a permutation
        ext = {}
        for p in all_perms:
            pref = p[:-1]
            ext.setdefault(pref, set()).add(p[-1])
        
        used = set()
        
        def is_dead_end(prefix):
            """Check if from this (n-1)-prefix we can extend to cover a new perm."""
            last = prefix[-1]
            for s in ext.get(prefix, ()):
                if s == last:
                    continue
                p = prefix + (s,)
                if p not in used:
                    return False
            return True
        
        # Start with first permutation
        word = list(range(1, n + 1))
        used.add(tuple(word))
        
        while len(used) < total:
            if time.time() > deadline:
                return None
            suffix = tuple(word[-(n - 1):])
            last = word[-1]
            
            # Candidates: symbols != last that complete a NEW permutation
            cands = []
            for s in ext.get(suffix, ()):
                if s == last:
                    continue
                p = suffix + (s,)
                if p not in used:
                    cands.append((s, p))
            
            if not cands:
                return None
            
            cands.sort()
            
            # Lookahead: prefer candidate whose new suffix is NOT a dead end
            chosen = None
            for s, p in cands:
                new_suffix = (suffix + (s,))[-(n - 1):]
                if not is_dead_end(new_suffix):
                    chosen = (s, p)
                    break
            
            if chosen is None:
                # All create dead ends; pick the one that leaves most options
                # (heuristic: pick one whose new suffix has most unused extensions)
                best_score = -1
                for s, p in cands:
                    new_suffix = (suffix + (s,))[-(n - 1):]
                    # Count how many unused perms this suffix can still reach
                    count = 0
                    for s2 in ext.get(new_suffix, ()):
                        if s2 != new_suffix[-1]:
                            p2 = new_suffix + (s2,)
                            if p2 not in used:
                                count += 1
                    if count > best_score:
                        best_score = count
                        chosen = (s, p)
                if chosen is None:
                    chosen = cands[0]
            
            s, p = chosen
            word.append(s)
            used.add(p)
        
        return word
    
    def _verify(word, n):
        """Check that all n! permutations appear as windows."""
        if len(word) < n:
            return False
        seen = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i + n])
            if len(set(w)) == n:  # only count valid permutations
                seen.add(w)
        return len(seen) == n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) * (n - 6) if n == 7 else len(seen) == len(list(permutations(range(1, n+1))))
    
    # Try greedy first
    result = None
    try:
        result = _greedy_superperm(n)
        if result is not None and len(result) > 0:
            # Verify
            if _verify(result, n):
                pass
            else:
                result = None
    except Exception:
        result = None
    
    if result is None:
        # Fallback to classic
        result = [int(c) for c in _classic(n)]
    
    # Final polish: try to remove redundant symbols
    # A symbol at position i (not first or last) can be removed if the window
    # that would be lost is still present elsewhere.
    # This is complex; for now, just return what we have.
    
    return "".join(map(str, result))
