import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    def generate_paley(p):
        """Generate Paley tournament on p vertices."""
        if p < 2:
            return None
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        n = p
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': n, 'orientation': orientation}

    def build_adj(n, orientation):
        """Build adjacency matrix from orientation bits."""
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        return adj

    def is_bad_4set(adj, a, b, c, d):
        """Check if the 4-set {a,b,c,d} has no dominator."""
        # A dominator exists if any vertex has out-degree 3 within the set.
        # Out-degree of a: adj[a,b] + adj[a,c] + adj[a,d]
        # Out-degree of b: adj[b,a] + adj[b,c] + adj[b,d]
        # ...
        
        # Note: adj[i,j] is 1 if i -> j.
        # For a 4-set, the sum of out-degrees is 6 (3 edges per vertex? No, 6 edges total, each contributes 1 to one out-degree).
        # So sum of out-degrees is 6.
        # If max out-degree is 3, it's good.
        # If max out-degree is 2, it's bad (since 2+2+1+1=6 or 2+2+2+0=6).
        
        oa = adj[a,b] + adj[a,c] + adj[a,d]
        if oa == 3: return False
        ob = adj[b,a] + adj[b,c] + adj[b,d]
        if ob == 3: return False
        oc = adj[c,a] + adj[c,b] + adj[c,d]
        if oc == 3: return False
        od = adj[d,a] + adj[d,b] + adj[d,c]
        if od == 3: return False
        
        return True

    def count_bad_sets(n, adj):
        """Count the number of bad 4-sets."""
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        if is_bad_4set(adj, a, b, c, d):
                            count += 1
        return count

    def local_search(n, time_limit):
        """
        Local search with incremental update of bad set count.
        Returns orientation list if valid, else None.
        """
        # Initialize random tournament
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        
        # Initial count
        bad_count = count_bad_sets(n, adj)
        
        if bad_count == 0:
            # Convert to orientation
            orientation = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] == 1:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return orientation

        start_ls = time.time()
        temperature = 1.0
        cooling = 0.999
        
        while time.time() - start_ls < time_limit:
            # Pick random edge (u, v)
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Determine current direction
            # If adj[u,v] == 1, u -> v. Else v -> u.
            current_dir = adj[u, v] # 1 if u->v, 0 if v->u (since adj[v,u] would be 1)
            
            # Calculate change in bad_count
            # Only 4-sets containing both u and v are affected.
            # These are sets {u, v, x, y} with x, y in V \ {u, v}, x < y.
            delta_bad = 0
            
            # We need to check each such set before and after flip.
            # To avoid modifying adj prematurely, we simulate the flip.
            # Or we can just calculate the contribution of u and v to the out-degrees of x and y?
            # Simpler: iterate x, y.
            
            for x in range(n):
                if x == u or x == v: continue
                for y in range(x+1, n):
                    if y == u or y == v: continue
                    
                    # Set S = {u, v, x, y}
                    # Check if S is bad BEFORE
                    # We need out-degrees of u, v, x, y within S.
                    
                    # This is getting complex to inline.
                    # Let's just call is_bad_4set on a modified view?
                    # No, let's compute out-degrees manually for the 4 nodes.
                    
                    # Current state:
                    # u->v? current_dir
                    # x->u? adj[x,u]
                    # x->v? adj[x,v]
                    # x->y? adj[x,y]
                    # y->u? adj[y,u]
                    # y->v? adj[y,v]
                    
                    # Out-degrees before:
                    # od_u = (1 if current_dir==1 else 0) + adj[u,x] + adj[u,y]
                    # Note: adj[u,x] is 1 if u->x.
                    # adj[x,u] is 1 if x->u.
                    # So adj[u,x] = 1 - adj[x,u].
                    
                    # Let's use adj matrix directly.
                    # Before flip:
                    # If current_dir == 1 (u->v):
                    #   out_u += 1 (for edge u-v)
                    #   out_v += 0
                    # Else (v->u):
                    #   out_u += 0
                    #   out_v += 1
                    
                    # The edges involving x and y are unchanged.
                    
                    # So:
                    # base_out_u = adj[u,x] + adj[u,y]
                    # base_out_v = adj[v,x] + adj[v,y]
                    # base_out_x = adj[x,u] + adj[x,v] + adj[x,y]
                    # base_out_y = adj[y,u] + adj[y,v] + adj[y,x]
                    
                    # Before:
                    # If u->v:
                    #   out_u = base_out_u + 1
                    #   out_v = base_out_v
                    # Else:
                    #   out_u = base_out_u
                    #   out_v = base_out_v + 1
                    
                    # After flip:
                    # If u->v becomes v->u:
                    #   out_u = base_out_u
                    #   out_v = base_out_v + 1
                    # Else:
                    #   out_u = base_out_u + 1
                    #   out_v = base_out_v
                    
                    # We can check if max(out) == 3 for before and after.
                    
                    base_out_u = adj[u,x] + adj[u,y]
                    base_out_v = adj[v,x] + adj[v,y]
                    base_out_x = adj[x,u] + adj[x,v] + adj[x,y]
                    base_out_y = adj[y,u] + adj[y,v] + adj[y,x]
                    
                    if current_dir == 1:
                        # Before: u->v
                        out_u_b = base_out_u + 1
                        out_v_b = base_out_v
                        # After: v->u
                        out_u_a = base_out_u
                        out_v_a = base_out_v + 1
                    else:
                        # Before: v->u
                        out_u_b = base_out_u
                        out_v_b = base_out_v + 1
                        # After: u->v
                        out_u_a = base_out_u + 1
                        out_v_a = base_out_v
                        
                    out_x = base_out_x
                    out_y = base_out_y
                    
                    # Check before
                    bad_before = False
                    if out_u_b < 3 and out_v_b < 3 and out_x < 3 and out_y < 3:
                        bad_before = True
                        
                    # Check after
                    bad_after = False
                    if out_u_a < 3 and out_v_a < 3 and out_x < 3 and out_y < 3:
                        bad_after = True
                        
                    if bad_before and not bad_after:
                        delta_bad -= 1
                    elif not bad_before and bad_after:
                        delta_bad += 1
            
            new_bad_count = bad_count + delta_bad
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_bad < 0:
                # Accept
                pass
            elif delta_bad == 0:
                # Accept with prob 0.5? Or always?
                if random.random() < 0.5:
                    pass
                else:
                    continue
            else:
                # Worse
                if random.random() < np.exp(-delta_bad / temperature):
                    pass
                else:
                    continue
            
            # Perform flip
            if current_dir == 1:
                adj[u, v] = 0
                adj[v, u] = 1
            else:
                adj[u, v] = 1
                adj[v, u] = 0
            
            bad_count = new_bad_count
            temperature *= cooling
            
            if bad_count == 0:
                orientation = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j] == 1:
                            orientation.append(0)
                        else:
                            orientation.append(1)
                return orientation

        return None

    # Fallback: Paley 67
    p67 = generate_paley(67)
    best_result = p67
    best_n = 67

    # Try to find smaller n
    # We iterate n from 31 to 66
    # Time budget per n: ~5 seconds
    # Total ~ 20 * 5 = 100s. Safe.
    
    for n in range(31, 67):
        if time.time() > deadline:
            break
        
        # Try a few restarts for each n?
        # Or just one long run?
        # Let's do one run of 5 seconds.
        
        # If n is prime, try Paley first (fast check)
        # But Paley only works for specific primes.
        # 31, 37, 41, 43, 47, 53, 59, 61, 67.
        
        # Check if n is a Paley prime
        # p = 3, 7 mod 8
        if n % 8 in [3, 7]:
            # Check if prime
            is_prime = True
            for i in range(2, int(n**0.5)+1):
                if n % i == 0:
                    is_prime = False
                    break
            if is_prime:
                res = generate_paley(n)
                # Verify
                adj = build_adj(n, res['orientation'])
                if count_bad_sets(n, adj) == 0:
                    return res
        
        # Local search
        # Use 4 seconds per n
        result = local_search(n, 4.0)
        if result is not None:
            return {'n': n, 'orientation': result}

    return best_result
