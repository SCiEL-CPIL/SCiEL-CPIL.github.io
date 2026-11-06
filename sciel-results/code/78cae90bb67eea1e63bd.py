"""Rewritten: bitset backtracking with C6 detection for 4-color C6-free K_n."""
import time
import random

def _has_c6_edge(color_masks, u, v, c):
    """Check if adding edge (u,v) in color c creates a C6.
    Returns True if a C6 would be created (i.e., there's a simple path of length 4 from u to v in color c).
    """
    # DFS from u, depth up to 4, checking if we can reach v at exactly depth 4
    # Path: u -> a -> b -> d -> e -> v, all in color c, all distinct
    def dfs(node, depth, visited):
        if depth == 4:
            return (color_masks[c][node] & (1 << v)) != 0
        if depth >= 4:
            return False
        neighbors = color_masks[c][node] & ~visited
        temp = neighbors
        while temp:
            lb = temp & (-temp)
            temp ^= lb
            idx = lb.bit_length() - 1
            if idx == v and depth < 3:
                continue
            visited |= (1 << idx)
            if dfs(idx, depth + 1, visited):
                return True
            visited &= ~(1 << idx)
        return False
    
    visited = (1 << u)
    return dfs(u, 0, visited)


def _check_full_c6(color_masks, n):
    """Check if any color class contains a C6. Returns True if valid (no C6)."""
    for c in range(4):
        for i in range(n):
            for j in range(i+1, n):
                if color_masks[c][i] & (1 << j):
                    old_i = color_masks[c][i]
                    old_j = color_masks[c][j]
                    color_masks[c][i] &= ~(1 << j)
                    color_masks[c][j] &= ~(1 << i)
                    if _has_c6_edge(color_masks, i, j, c):
                        color_masks[c][i] = old_i
                        color_masks[c][j] = old_j
                        return False
                    color_masks[c][i] = old_i
                    color_masks[c][j] = old_j
    return True


def _edge_to_pair(n, edge_idx):
    """Map edge index to (i, j) pair in lexicographic order."""
    i = 0
    offset = 0
    while True:
        row_len = n - 1 - i
        if edge_idx < offset + row_len:
            j = i + 1 + (edge_idx - offset)
            return (i, j)
        offset += row_len
        i += 1
        if i >= n:
            break
    return (0, 1)


def _greedy_init(n):
    """Greedy initialization."""
    color_masks = [[0] * n for _ in range(4)]
    for i in range(n):
        for j in range(i+1, n):
            placed = False
            for c in range(4):
                if not _has_c6_edge(color_masks, i, j, c):
                    color_masks[c][i] |= (1 << j)
                    color_masks[c][j] |= (1 << i)
                    placed = True
                    break
            if not placed:
                c = random.randint(0, 3)
                color_masks[c][i] |= (1 << j)
                color_masks[c][j] |= (1 << i)
    return color_masks


def _local_search(color_masks, n, deadline, max_iters=5000):
    """Local search to fix C6 violations."""
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        violations = []
        for c in range(4):
            for i in range(n):
                for j in range(i+1, n):
                    if color_masks[c][i] & (1 << j):
                        if _has_c6_edge(color_masks, i, j, c):
                            violations.append((i, j, c))
        
        if not violations:
            return True
        
        i, j, c = random.choice(violations)
        color_masks[c][i] &= ~(1 << j)
        color_masks[c][j] &= ~(1 << i)
        
        placed = False
        for c2 in range(4):
            if c2 == c:
                continue
            if not _has_c6_edge(color_masks, i, j, c2):
                color_masks[c2][i] |= (1 << j)
                color_masks[c2][j] |= (1 << i)
                placed = True
                break
        if not placed:
            color_masks[c][i] |= (1 << j)
            color_masks[c][j] |= (1 << i)
    
    return _check_full_c6(color_masks, n)


def _backtrack(n, edge_idx, color_masks, deadline, best_result):
    """Backtracking search."""
    if time.time() > deadline:
        return False
    
    if edge_idx == n * (n - 1) // 2:
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                for c in range(4):
                    if color_masks[c][i] & (1 << j):
                        colors.append(c)
                        break
        best_result['n'] = n
        best_result['colors'] = colors
        return True
    
    i, j = _edge_to_pair(n, edge_idx)
    
    for c in range(4):
        if not _has_c6_edge(color_masks, i, j, c):
            color_masks[c][i] |= (1 << j)
            color_masks[c][j] |= (1 << i)
            
            if _backtrack(n, edge_idx + 1, color_masks, deadline, best_result):
                return True
            
            color_masks[c][i] &= ~(1 << j)
            color_masks[c][j] &= ~(1 << i)
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Seed solution: n=9
    best_result = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        # Try greedy + local search
        for attempt in range(30):
            if time.time() > deadline:
                break
            color_masks = _greedy_init(n)
            if _local_search(color_masks, n, deadline, max_iters=3000):
                colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        for c in range(4):
                            if color_masks[c][i] & (1 << j):
                                colors.append(c)
                                break
                best_result = {'n': n, 'colors': colors}
                break
        
        if best_result['n'] == n:
            continue
        
        # Backtracking for smaller n
        if n <= 15:
            color_masks = [[0] * n for _ in range(4)]
            if _backtrack(n, 0, color_masks, deadline, best_result):
                if best_result['n'] == n:
                    continue
        
        if n > 13:
            break
    
    return best_result
