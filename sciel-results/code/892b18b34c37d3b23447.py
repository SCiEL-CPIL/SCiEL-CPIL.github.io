import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 31
    target_edges = 91
    q = 5

    # 1. Construct the Polarity Graph of PG(2,5)
    # Points of PG(2,5) are non-zero triples (x,y,z) in F_5^3 modulo scaling.
    # Number of points = (5^3-1)/(5-1) = 124/4 = 31.
    # Standard representation:
    # 0..24: (x, y, 1) for x, y in 0..4
    # 25..29: (x, 1, 0) for x in 0..4
    # 30: (1, 0, 0)

    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    assert len(points) == 31

    # Define adjacency based on the polarity B(P, Q) = x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1

    # The polarity graph of PG(2,5) is C4-free and has 90 edges.
    if len(edges) != 90:
        # Fallback: if construction fails (shouldn't happen), return a random graph
        # to ensure validity, though score will be poor.
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return [list(e) for e in all_pairs[:91]]

    # 2. Add one edge to reach 91 edges.
    # We want to choose the edge that minimizes the number of C4s created.
    # Adding edge (u, v) creates C4s of the form u-v-x-w-u.
    # This corresponds to paths of length 3 from u to v in the existing graph.
    # Number of new C4s = number of common neighbors of u and v? No.
    # A C4 containing the new edge (u,v) is u-v-x-w-u.
    # This requires (v,x), (x,w), (w,u) to be edges.
    # This is equivalent to the number of paths of length 3 from u to v.
    # Let A be the adjacency matrix.
    # Number of paths of length 3 from u to v is (A^3)[u, v].
    # So we want to minimize (A^3)[u, v] for non-edges (u, v).

    A = adj
    A2 = np.dot(A, A)
    A3 = np.dot(A2, A)

    best_edge = None
    min_c4s = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                c4s_created = A3[i, j]
                if c4s_created < min_c4s:
                    min_c4s = c4s_created
                    best_edge = (i, j)

    if best_edge:
        u, v = best_edge
        edges.append(best_edge)
        adj[u, v] = 1
        adj[v, u] = 1

    # 3. Local Search / Simulated Annealing to minimize C4s further.
    # It is possible that a swap of edges can reduce the C4 count below min_c4s.
    # The minimum possible C4 count for 91 edges on 31 vertices is likely 1.
    # If min_c4s is 1, we are likely optimal.
    # If min_c4s > 1, we should try to improve.

    def count_c4(adj_mat):
        # Vectorized C4 count
        A2 = np.dot(adj_mat, adj_mat)
        np.fill_diagonal(A2, 0)
        # Sum of C(k, 2) for all u < v
        # k = A2[u, v]
        # C(k, 2) = k * (k - 1) / 2
        # Sum over u < v
        # Since A2 is symmetric, sum_{u < v} f(A2[u, v]) = 0.5 * (sum_{u != v} f(A2[u, v]))
        # sum_{u != v} k(k-1)
        # Then C4s = 0.5 * Sum_{u < v} C(k, 2) = 0.5 * Sum_{u < v} k(k-1)/2 = 0.25 * Sum_{u < v} k(k-1)
        # Sum_{u < v} k(k-1) = 0.5 * Sum_{u != v} k(k-1)
        # So C4s = 0.125 * Sum_{u != v} k(k-1)
        # Let's verify with 1 C4.
        # Two pairs have k=2. k(k-1)=2. Sum = 4.
        # 0.125 * 4 = 0.5? No.
        # My previous derivation:
        # C4s = 0.5 * Sum_{u < v} C(k, 2)
        # Sum_{u < v} C(k, 2) = Sum_{u < v} k(k-1)/2
        # For 1 C4, two pairs have k=2. C(2,2)=1. Sum = 2.
        # C4s = 0.5 * 2 = 1. Correct.
        # So C4s = 0.5 * Sum_{u < v} k(k-1)/2 = 0.25 * Sum_{u < v} k(k-1)
        # Sum_{u < v} k(k-1) = 0.5 * Sum_{u != v} k(k-1)
        # C4s = 0.125 * Sum_{u != v} k(k-1)
        # Wait. 0.25 * (0.5 * Sum_{u != v}) = 0.125 * Sum_{u != v}.
        # For 1 C4: Sum_{u != v} k(k-1) = 4.
        # 0.125 * 4 = 0.5. Still wrong.
        # Let's re-evaluate.
        # Sum_{u < v} C(k, 2) = 2.
        # C4s = Sum / 2 = 1.
        # Sum_{u < v} k(k-1) = 2 * Sum_{u < v} C(k, 2) = 4.
        # C4s = 0.25 * Sum_{u < v} k(k-1) = 0.25 * 4 = 1. Correct.
        # So we need Sum_{u < v} k(k-1).
        # Sum_{u < v} k(k-1) = 0.5 * Sum_{u != v} k(k-1)
        # So C4s = 0.25 * 0.5 * Sum_{u != v} k(k-1) = 0.125 * Sum_{u != v} k(k-1).
        # Why did I get 0.5 before?
        # Ah, Sum_{u != v} k(k-1) for 1 C4 is 4.
        # 0.125 * 4 = 0.5.
        # There is a factor of 2 missing somewhere.
        # Let's just compute Sum_{u < v} directly to be safe.
        
        # Upper triangular part
        # mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # k_vals = A2[mask]
        # sum_val = np.sum(k_vals * (k_vals - 1))
        # return sum_val // 4
        
        # Let's stick to the safe integer arithmetic.
        # Sum_{u < v} k(k-1)
        # We can compute this by summing the upper triangle of A2*(A2-1)
        # and dividing by 4.
        
        # A2 is integer.
        # prod = A2 * (A2 - 1)
        # sum_upper = np.sum(np.triu(prod, 1))
        # return sum_upper // 4
        
        # Let's test this logic.
        # 1 C4. Two pairs (u,w) and (v,x) have k=2.
        # prod[u,w] = 2*1 = 2.
        # prod[v,x] = 2*1 = 2.
        # sum_upper = 4.
        # return 4 // 4 = 1. Correct.
        
        prod = A2 * (A2 - 1)
        sum_upper = np.sum(np.triu(prod, 1))
        return sum_upper // 4

    current_score = count_c4(adj)
    best_score = current_score
    best_adj = adj.copy()
    best_edges = edges.copy()

    # If we already have 1 C4, it's likely optimal.
    if current_score == 1:
        # Just return the best edges found so far.
        # No need for SA if we hit the lower bound (likely 1).
        # But let's run a quick check to be sure.
        pass

    # Simulated Annealing
    # Move: Swap an edge (u,v) with a non-edge (a,b).
    # Keep edge count constant at 91.
    
    temp = 1.0
    cooling_rate = 0.999
    
    # Precompute list of edges and non-edges?
    # Or just sample.
    
    while time.time() < deadline - 5:
        # Pick random edge to remove
        # Sample from edges list
        e_idx = random.randint(0, len(edges) - 1)
        u, v = edges[e_idx]
        
        # Pick random non-edge to add
        # Sample random pair, check if non-edge
        found = False
        attempts = 0
        while not found and attempts < 20:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a != b and adj[a, b] == 0:
                # Ensure we don't swap (u,v) with (u,v)
                if (a == u and b == v) or (a == v and b == u):
                    attempts += 1
                    continue
                found = True
            attempts += 1
        
        if not found:
            continue

        # Apply move
        adj[u, v] = 0
        adj[v, u] = 0
        adj[a, b] = 1
        adj[b, a] = 1
        
        # Update edges list
        edges[e_idx] = (a, b)
        
        new_score = count_c4(adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
                best_edges = edges.copy()
        else:
            delta = new_score - current_score
            if random.random() < np.exp(-delta / temp):
                current_score = new_score
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                edges[e_idx] = (u, v)
        
        temp *= cooling_rate
        if temp < 0.01:
            temp = 0.01

    # Return the best edges found
    # best_edges is a list of tuples. Convert to list of lists.
    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(f"Generated {len(edges)} edges")
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A2 = np.dot(adj, adj)
    np.fill_diagonal(A2, 0)
    prod = A2 * (A2 - 1)
    sum_upper = np.sum(np.triu(prod, 1))
    c4s = sum_upper // 4
    print(f"C4 count: {c4s}")
