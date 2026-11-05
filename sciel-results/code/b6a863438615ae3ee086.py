import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph with non-abelian groups for k=3.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3:
        return _construct_petersen_lift_non_abelian(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift_non_abelian(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in non-abelian groups.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    # Store as list of edge indices with signs (+1 for u->v, -1 for v->u)
    walks = []
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [])]
        
        while stack:
            v, prev_v, l, path = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    sign = 1
                else:
                    sign = -1
                
                new_path = path + [(e_idx, sign)]
                new_l = l + 1
                
                if next_v == start_v and new_l > 0:
                    walks.append(new_path)
                
                stack.append((next_v, v, new_l, new_path))
    
    if not walks:
        return _construct_luw(3, g, d=7)

    # Define groups to try
    # Group 1: SL(2, 3) - Order 24
    # Group 2: PSL(2, 5) - Order 60
    # Group 3: SL(2, 5) - Order 120
    
    def get_group_elements(group_id):
        if group_id == 1: # SL(2,3)
            elements = []
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        # ad - bc = 1
                        # d = (1 + bc) * a^-1
                        if a == 0:
                            continue
                        # a is 1 or 2. 1^-1=1, 2^-1=2
                        inv_a = 1 if a == 1 else 2
                        d = (1 + b*c) * inv_a % 3
                        elements.append(((a, b), (c, d)))
            return elements
        elif group_id == 2: # PSL(2,5) ~ A5
            # We can use permutations of 5 elements
            # Generate all even permutations
            import itertools
            perms = list(itertools.permutations(range(5)))
            even_perms = []
            for p in perms:
                # Check if even
                inv_count = 0
                for i in range(5):
                    for j in range(i+1, 5):
                        if p[i] > p[j]:
                            inv_count += 1
                if inv_count % 2 == 0:
                    even_perms.append(p)
            return even_perms
        elif group_id == 3: # SL(2,5)
            elements = []
            for a in range(5):
                for b in range(5):
                    for c in range(5):
                        if a == 0:
                            continue
                        # Find d such that ad - bc = 1
                        # ad = 1 + bc
                        # d = (1 + bc) * a^-1
                        inv_a = pow(a, -1, 5)
                        d = (1 + b*c) * inv_a % 5
                        elements.append(((a, b), (c, d)))
            return elements

    def multiply_elements(g1, g2, group_id):
        if group_id == 1:
            # Matrix mult mod 3
            a1, b1 = g1[0]
            c1, d1 = g1[1]
            a2, b2 = g2[0]
            c2, d2 = g2[1]
            a = (a1*a2 + b1*c2) % 3
            b = (a1*b2 + b1*d2) % 3
            c = (c1*a2 + d1*c2) % 3
            d = (c1*b2 + d1*d2) % 3
            return ((a, b), (c, d))
        elif group_id == 2:
            # Permutation composition
            # (g1 o g2)[i] = g1[g2[i]]
            return tuple(g1[g2[i]] for i in range(5))
        elif group_id == 3:
            # Matrix mult mod 5
            a1, b1 = g1[0]
            c1, d1 = g1[1]
            a2, b2 = g2[0]
            c2, d2 = g2[1]
            a = (a1*a2 + b1*c2) % 5
            b = (a1*b2 + b1*d2) % 5
            c = (c1*a2 + d1*c2) % 5
            d = (c1*b2 + d1*d2) % 5
            return ((a, b), (c, d))

    def identity_element(group_id):
        if group_id == 1:
            return ((1, 0), (0, 1))
        elif group_id == 2:
            return (0, 1, 2, 3, 4)
        elif group_id == 3:
            return ((1, 0), (0, 1))

    def invert_element(g, group_id):
        if group_id == 1:
            a, b = g[0]
            c, d = g[1]
            # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
            return ((d, -b%3), (-c%3, a))
        elif group_id == 2:
            inv = [0]*5
            for i in range(5):
                inv[g[i]] = i
            return tuple(inv)
        elif group_id == 3:
            a, b = g[0]
            c, d = g[1]
            return ((d, -b%5), (-c%5, a))

    # Try groups in order of size (smaller first)
    groups_to_try = [1, 2, 3]
    
    for gid in groups_to_try:
        if time.time() > deadline:
            break
            
        elements = get_group_elements(gid)
        n_elems = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        id_elem = identity_element(gid)
        
        # Precompute multiplication table
        mult_table = np.zeros((n_elems, n_elems), dtype=np.int64)
        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements):
                prod = multiply_elements(e1, e2, gid)
                mult_table[i, j] = elem_to_idx[prod]
        
        # Precompute inverse table
        inv_table = np.zeros(n_elems, dtype=np.int64)
        for i, e in enumerate(elements):
            inv_e = invert_element(e, gid)
            inv_table[i] = elem_to_idx[inv_e]
            
        # Random search for voltages
        # We assign a voltage to each edge.
        # The voltage is an index in 0..n_elems-1.
        # For a walk, the product is the composition of voltages.
        # If the edge is traversed in reverse, we use the inverse voltage.
        
        found = False
        for _ in range(500): # Try 500 random assignments
            if time.time() > deadline:
                break
                
            voltages = [random.randint(0, n_elems - 1) for _ in range(15)]
            
            valid = True
            for walk in walks:
                # Compute product of voltages
                prod_idx = elem_to_idx[id_elem]
                for e_idx, sign in walk:
                    v_idx = voltages[e_idx]
                    if sign == -1:
                        v_idx = inv_table[v_idx]
                    prod_idx = mult_table[prod_idx, v_idx]
                
                if prod_idx == elem_to_idx[id_elem]:
                    valid = False
                    break
            
            if valid:
                # Build the lifted graph
                edges = _build_lifted_graph_non_abelian(pet_edges, voltages, elements, gid)
                return 10 * n_elems, edges
                
    # Fallback to LUW
    return _construct_luw(3, g, d=7)

def _build_lifted_graph_non_abelian(pet_edges, voltages, elements, gid):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute inverses
    inv_table = {}
    for i, e in enumerate(elements):
        inv_e = invert_element(e, gid)
        inv_table[i] = elements.index(inv_e)
        
    # Precompute multiplication table
    n_elems = len(elements)
    mult_table = {}
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            prod = multiply_elements(e1, e2, gid)
            mult_table[(i, j)] = elements.index(prod)

    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol_idx = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
                
            if sign == -1:
                vol_idx = inv_table[vol_idx]
            
            for layer in range(n_elems):
                u_node = layer * n_base + u
                # target_layer = layer * vol
                # In group terms: target = layer * vol
                # If layer is index, we need to find index of layer * vol
                # This is mult_table[(layer, vol_idx)]
                target_layer = mult_table[(layer, vol_idx)]
                v_node = target_layer * n_base + v
                
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d.
    """
    if d is None:
        d = max(3, (g - 2) // 2)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
