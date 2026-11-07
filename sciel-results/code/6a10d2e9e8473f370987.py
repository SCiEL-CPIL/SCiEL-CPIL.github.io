"""Direct construction of 3-coloring for 5x10 toroidal grid with bicolored forest constraint."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_EDGES = P['edges']
_N = P['n']
_ROWS = 5
_COLS = 10

_ADJ = defaultdict(list)
for u, v in _EDGES:
    _ADJ[u].append(v)
    _ADJ[v].append(u)


def _bicolored_cycle_check(coloring, k):
    """Check that for every pair of colors (a,b) with a<b<k, the subgraph
    induced by vertices with color in {a,b} is a forest (no cycles)."""
    for a in range(k):
        for b in range(a + 1, k):
            # Collect vertices with color a or b
            verts = [i for i in range(_N) if coloring[i] == a or coloring[i] == b]
            if len(verts) < 3:
                continue
            vset = set(verts)
            # Use union-find to detect cycles
            parent = {v: v for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                parent[rx] = ry
                return True
            
            for u, v in _EDGES:
                if u in vset and v in vset:
                    if not union(u, v):
                        return False
    return True


def _direct_3coloring():
    """Direct construction of a 3-coloring for the 5x10 toroidal grid.
    
    Vertex v: row = v // 10, col = v % 10
    Try pattern: color = (row + 2*col) % 3
    """
    coloring = [0] * _N
    for v in range(_N):
        row = v // _COLS
        col = v % _COLS
        coloring[v] = (row + 2 * col) % 3
    return coloring


def _try_3color_patterns():
    """Try multiple 3-coloring patterns for the 5x10 toroidal grid and return
    the first one that satisfies the bicolored forest constraint."""
    # Various linear patterns: color = (a*row + b*col) % 3
    # We need: for each pair of colors, the induced subgraph is a forest
    patterns = []
    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue
            patterns.append((a, b))
    
    for a, b in patterns:
        coloring = [0] * _N
        for v in range(_N):
            row = v // _COLS
            col = v % _COLS
            coloring[v] = (a * row + b * col) % 3
        # Check adjacency constraint (no two adjacent vertices same color)
        valid_adj = True
        for u, v in _EDGES:
            if coloring[u] == coloring[v]:
                valid_adj = False
                break
        if not valid_adj:
            continue
        # Check bicolored forest constraint
        if _bicolored_cycle_check(coloring, 3):
            return coloring
    return None


def _backtrack_3coloring(deadline):
    """Backtracking search for a valid 3-coloring if direct construction fails."""
    coloring = [-1] * _N
    order = list(range(_N))
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y, parent):
        rx, ry = find(x, parent), find(y, parent)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    def check_partial(assigned):
        """Check if partial assignment violates bicolored forest constraint."""
        for a in range(3):
            for b in range(a + 1, 3):
                verts = [i for i in assigned if coloring[i] in (a, b)]
                if len(verts) < 3:
                    continue
                vset = set(verts)
                parent = {v: v for v in vset}
                for u, v in _EDGES:
                    if u in vset and v in vset:
                        if not union(u, v, parent):
                            return False
        return True
    
    def backtrack(pos, assigned):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        v = order[pos]
        for c in range(3):
            # Check adjacency
            ok = True
            for nb in _ADJ[v]:
                if nb in assigned and coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            coloring[v] = c
            assigned.add(v)
            if check_partial(assigned):
                if backtrack(pos + 1, assigned):
                    return True
            coloring[v] = -1
            assigned.discard(v)
        return False
    
    assigned = set()
    if backtrack(0, assigned):
        return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    # First: try direct 3-coloring patterns (fast, O(1) per pattern)
    coloring = _try_3color_patterns()
    if coloring is not None:
        return [int(c) for c in coloring]

    # Second: try direct single-pattern 3-coloring
    coloring = _direct_3coloring()
    if coloring is not None and _bicolored_cycle_check(coloring, 3):
        return [int(c) for c in coloring]

    # Third: backtracking for 3-coloring
    if time.time() < deadline:
        coloring = _backtrack_3coloring(deadline)
        if coloring is not None:
            return [int(c) for c in coloring]

    # Fourth: try 4-coloring via backtracking
    if time.time() < deadline:
        coloring = [-1] * _N
        order = list(range(_N))
        
        def find4(x, parent):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union4(x, y, parent):
            rx, ry = find4(x, parent), find4(y, parent)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        def check_partial4(assigned):
            for a in range(4):
                for b in range(a + 1, 4):
                    verts = [i for i in assigned if coloring[i] in (a, b)]
                    if len(verts) < 3:
                        continue
                    vset = set(verts)
                    parent = {v: v for v in vset}
                    for u, v in _EDGES:
                        if u in vset and v in vset:
                            if not union4(u, v, parent):
                                return False
            return True
        
        def backtrack4(pos, assigned):
            if time.time() > deadline:
                return False
            if pos == _N:
                return True
            v = order[pos]
            for c in range(4):
                ok = True
                for nb in _ADJ[v]:
                    if nb in assigned and coloring[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                coloring[v] = c
                assigned.add(v)
                if check_partial4(assigned):
                    if backtrack4(pos + 1, assigned):
                        return True
                coloring[v] = -1
                assigned.discard(v)
            return False
        
        assigned = set()
        if backtrack4(0, assigned):
            return [int(c) for c in coloring]

    # Fallback: 50 colors (always valid)
    return list(range(_N))
# EVOLVE-BLOCK-END
