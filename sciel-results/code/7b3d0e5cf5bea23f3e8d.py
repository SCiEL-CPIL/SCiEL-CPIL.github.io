import time
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Parse matrix
    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])
    
    if n == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    # Represent each row as a frozenset of input indices
    rows = []
    for i in range(m):
        support = frozenset(j for j in range(n) if matrix[i][j] % 2 == 1)
        rows.append(support)
    
    # CSE-based synthesis
    wire_map = {}
    for j in range(n):
        wire_map[frozenset([j])] = j
    wire_map[frozenset()] = n
    
    next_wire = n + 1
    gates = []
    
    sys.setrecursionlimit(10000)
    
    def get_wire(support):
        nonlocal next_wire
        if time.time() > deadline:
            return n
        
        if support in wire_map:
            return wire_map[support]
        if len(support) == 0:
            return n
        if len(support) == 1:
            j = next(iter(support))
            return j
        
        items = sorted(support)
        k = len(items)
        
        best_left = None
        best_right = None
        best_score = -1
        
        # Try balanced split
        mid = k // 2
        left = frozenset(items[:mid])
        right = frozenset(items[mid:])
        score = (1 if left in wire_map else 0) + (1 if right in wire_map else 0)
        if score > best_score:
            best_score = score
            best_left = left
            best_right = right
        
        # Try other splits if k is small
        if k <= 8:
            for mid2 in range(1, k):
                if mid2 == mid:
                    continue
                left2 = frozenset(items[:mid2])
                right2 = frozenset(items[mid2:])
                score2 = (1 if left2 in wire_map else 0) + (1 if right2 in wire_map else 0)
                if score2 > best_score:
                    best_score = score2
                    best_left = left2
                    best_right = right2
        
        left, right = best_left, best_right
        wire_left = get_wire(left)
        wire_right = get_wire(right)
        
        gates.append([wire_left, wire_right])
        wire = next_wire
        next_wire += 1
        wire_map[support] = wire
        return wire
    
    outputs = []
    for i in range(m):
        support = rows[i]
        wire = get_wire(support)
        outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}
