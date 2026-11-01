import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Fallback: LUW graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try to find a Cayley graph on PSL(2, q)
    def try_cayley(q):
        if time.time() > deadline:
            return None
            
        # Build PSL(2, q)
        # Elements are matrices [[a, b], [c, d]] with ad-bc = 1 mod q
        # We work in GF(q)
        
        # Generate all elements
        elements = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # ad - bc = 1 => ad = 1 + bc
                    # If a=0, -bc=1 => bc = -1.
                    # If a!=0, d = (1+bc)/a.
                    if a == 0:
                        # -bc = 1 => bc = -1 mod q
                        # b cannot be 0.
                        for b in range(1, q):
                            # c = -1 * b^{-1}
                            # Find inverse of b
                            inv_b = pow(b, q-2, q)
                            c = (-1 * inv_b) % q
                            d = 0
                            # Check if determinant is 1
                            # 0*d - b*c = -bc = 1
                            elements.append((0, b, c, 0))
                    else:
                        for b in range(q):
                            for c in range(q):
                                # d = (1 + bc) * a^{-1}
                                inv_a = pow(a, q-2, q)
                                d = ((1 + b*c) * inv_a) % q
                                # Check determinant
                                if (a*d - b*c) % q == 1:
                                    elements.append((a, b, c, d))
        
        # Remove duplicates and identify PSL (mod center)
        # In PSL(2, q), M ~ -M.
        # We can just use the list of unique matrices, but we need to be careful with the group structure.
        # Actually, for q=11, 13, the center is {I, -I}.
        # We can just work with SL(2, q) and ensure the generating set is symmetric and doesn't contain -I?
        # No, PSL is better.
        
        # Let's just use the list of elements in SL(2, q) and treat -M as equivalent to M?
        # No, simpler: Just use the elements of SL(2, q) and if we find a graph, it's a cover.
        # But we want the smallest n.
        
        # Let's just use the unique matrices in SL(2, q).
        unique_elements = list(set(elements))
        n = len(unique_elements)
        
        # Map element to index
        elem_to_idx = {e: i for i, e in enumerate(unique_elements)}
        
        # Define multiplication
        def mul_mat(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return (
                (a1*a2 + b1*c2) % q,
                (a1*b2 + b1*d2) % q,
                (c1*a2 + d1*c2) % q,
                (c1*b2 + d1*d2) % q
            )
        
        # Precompute multiplication table? Too big (660^2).
        # Just compute on the fly.
        
        # Find involutions (order 2)
        involutions = []
        for e in unique_elements:
            # e^2 = I
            e2 = mul_mat(e, e)
            if e2 == (1, 0, 0, 1):
                involutions.append(e)
        
        if len(involutions) < 1:
            return None
            
        # We need a set S of 5 elements.
        # S must be symmetric.
        # So S = {a, a^-1, b, b^-1, c} where c is an involution.
        # a, b must not be involutions.
        
        non_inv = [e for e in unique_elements if e != (1,0,0,1) and mul_mat(e,e) != (1,0,0,1)]
        
        # Try random sets
        for _ in range(2000):
            if time.time() > deadline:
                return None
            if not non_inv:
                break
            a = random.choice(non_inv)
            b = random.choice(non_inv)
            if a == b or a == mul_mat(b, b): # b^-1
                continue
            c = random.choice(involutions)
            
            # Check if a^-1 is in the set?
            # We define S = {a, inv(a), b, inv(b), c}
            # We need to compute inv(a).
            def inv_mat(m):
                a, b, c, d = m
                # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
                return (d, (-b)%q, (-c)%q, a)
            
            a_inv = inv_mat(a)
            b_inv = inv_mat(b)
            
            S = [a, a_inv, b, b_inv, c]
            
            # Ensure S does not contain Identity
            if (1,0,0,1) in S:
                continue
                
            # Check if S has duplicates
            if len(set(S)) != 5:
                continue
                
            # Check girth >= 9
            # BFS from Identity (1,0,0,1)
            # We need to visit 426 unique nodes in 4 steps.
            # Actually, we just need to ensure no collisions in the ball of radius 4.
            
            start = (1, 0, 0, 1)
            # visited: dict (node, last_step) -> step
            # Or just a set of nodes visited in <= 4 steps.
            # And ensure that the path is a tree.
            
            # BFS
            # Queue: (node, depth, parent)
            # We stop if we find a node already visited at depth < 4?
            # No, if we find a node visited at depth d1 and we are at depth d2, and d1+d2 < 9?
            # Actually, if the ball of radius 4 is a tree, there are no cycles of length <= 8.
            # So we just check if the number of unique nodes visited in 4 steps is 426.
            
            visited = {start: 0}
            queue = deque([(start, 0)])
            count = 1
            
            while queue:
                node, depth = queue.popleft()
                if depth == 4:
                    continue
                for s in S:
                    next_node = mul_mat(node, s)
                    if next_node not in visited:
                        visited[next_node] = depth + 1
                        count += 1
                        queue.append((next_node, depth + 1))
            
            if count == 426:
                # We found a valid generating set!
                # Now construct the graph
                # Vertices are the elements of the group.
                # Edges: (u, u*s) for all u, s in S.
                # We only add edge if u < u*s to avoid duplicates?
                # Or just return all edges?
                # The task asks for a list of edges.
                
                edges = []
                for u in unique_elements:
                    idx_u = elem_to_idx[u]
                    for s in S:
                        v = mul_mat(u, s)
                        idx_v = elem_to_idx[v]
                        if idx_u < idx_v:
                            edges.append((idx_u, idx_v))
                
                return n, edges
        
        return None

    # Try PSL(2, 11)
    result = try_cayley(11)
    if result:
        return result
        
    # Try PSL(2, 13)
    result = try_cayley(13)
    if result:
        return result

    # Fallback to LUW
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
