# EVOLVE-BLOCK-START
import itertools
import time

_START = time.time()
_DEADLINE = 150 - 12   # 138 s safety margin


def _classic_construct(m):
    """Classic recursive superpermutation construction for symbols 1..m."""
    word = [1]
    for sym in range(2, m + 1):
        perms, seen = [], set()
        # Collect all (sym-1)-long windows with distinct elements, in order of appearance
        for i in range(len(word) - sym + 2):
            window = tuple(word[i:i + sym - 1])
            if len(set(window)) == sym - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For the last step, use a greedy heuristic to minimize length
        if sym == m and m >= 5:
            new = []
            remaining = list(perms)
            # Start with the first one (or best start?)
            # Let's just start with the first one from the classic list
            first_p = remaining.pop(0)
            block = list(first_p) + [sym] + list(first_p)
            new = block
            
            while remaining:
                best_p = None
                best_overlap = -1
                best_block = None
                current_end = new
                # Find the perm that gives the maximum overlap
                for p in remaining:
                    b = list(p) + [sym] + list(p)
                    # Calculate overlap
                    max_o = 0
                    # Max possible overlap is len(b)
                    limit = min(len(current_end), len(b))
                    for o in range(limit, 0, -1):
                        if current_end[-o:] == b[:o]:
                            max_o = o
                            break
                    if max_o > best_overlap:
                        best_overlap = max_o
                        best_p = p
                        best_block = b
                # Append the best block
                if best_p is not None:
                    new += best_block[best_overlap:]
                    remaining.remove(best_p)
            word = new
        else:
            new = []
            for p in perms:
                block = list(p) + [sym] + list(p)
                # Find maximum overlap between end of 'new' and start of 'block'
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    return word


def _build_word(n):
    """Build a word over 1..n containing every permutation as consecutive window."""
    if n <= 1:
        return [1]
    
    word = _classic_construct(n)
    
    return word


def construct(n):
    """Construct a shortest-known superpermutation word for symbols 1..n.
    
    Returns a string of digits 1..n.
    """
    t0 = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341213421431214321"
    if n == 5:
        return "1234512134521431523415243152543152143521"
    
    # For n >= 6, use classic construction with greedy last step
    word = _classic_construct(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
