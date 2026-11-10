import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    
    # 1. Precompute 3-APs and 4-APs
    three_aps = []
    three_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            three_aps_by_v[a].append(mask)
            three_aps_by_v[b].append(mask)
            three_aps_by_v[c].append(mask)
            
    four_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[a].append(mask)
            four_aps_by_v[b].append(mask)
            four_aps_by_v[c].append(mask)
            four_aps_by_v[e].append(mask)
            
    # 2. Salem-type structured initial set
    # Use quadratic residues mod p, mapped to [1,N]
    # For p=251 (prime), quadratic residues give a set with no 4-APs
    # and good 3-AP density
    
    def salem_set():
        """Generate a Salem-type set using quadratic residues."""
        p = 251  # Prime close to N
        # Quadratic residues mod p
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        # Map to [1, N] by taking residues that are <= N
        # Actually, let's use a different approach: use the set {x^2 mod p : x=1..p-1}
        # and map each residue r to position r (if r >= 1)
        # But we need positions in [1, 256], and residues are in [0, 250]
        # So map residue r (r>=1) to position r
        s = set()
        for r in residues:
            if 1 <= r <= N:
                s.add(r)
        return s
    
    # Alternative: use a multiplicative subgroup approach
    # Or use a simpler structured set: numbers with even binary digit count
    
    def structured_set_v1():
        """Set of numbers in [1,N] with even popcount - tends to avoid 4-APs."""
        s = set()
        for i in range(1, N+1):
            if bin(i).count('1') % 2 == 0:
                s.add(i)
        return s
    
    def structured_set_v2():
        """Quadratic residues mod 251, mapped to [1,256]."""
        p = 251
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        # Filter to [1, 256]
        s = set()
        for r in residues:
            if 1 <= r <= N:
                s.add(r)
        return s
    
    def structured_set_v3():
        """Use a known 4-AP-free construction: {i : i mod 4 in {0,1}} filtered."""
        # This is too dense, will have 4-APs
        # Instead, use a greedy construction but with 3-AP awareness
        s = set()
        for x in range(1, N+1):
            # Check if adding x creates a 4-AP
            valid = True
            for ap_mask in four_aps_by_v[x]:
                other = ap_mask & ~(1 << (x-1))
                if other and (all((other >> (i-1)) & 1 for i in range(1, N+1) if (other >> (i-1)) & 1) is False):
                    pass
                # Simpler: check if all other elements of any 4-AP containing x are in s
                if (ap_mask & ~(1 << (x-1))) and all((ap_mask >> (i-1)) & 1 for i in range(1, N+1) if (ap_mask >> (i-1)) & 1 and i != x):
                    valid = False
                    break
            if valid:
                s.add(x)
        return s
    
    # Use Salem-type set as initial
    initial_set = structured_set_v2()
    
    # Verify it's 4-AP-free; if not, remove violating elements
    def is_4ap_free(s):
        s_mask = 0
        for x in s:
            s_mask |= (1 << (x-1))
        for d in range(1, N // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if a in s and b in s and c in s and e in s:
                    return False
        return True
    
    if not is_4ap_free(initial_set):
        # Remove elements to make it 4-AP-free (greedy removal)
        for x in sorted(initial_set):
            if is_4ap_free(initial_set):
                break
            initial_set.discard(x)
    
    # Convert to mask
    current_mask = 0
    for x in initial_set:
        current_mask |= (1 << (x-1))
    
    # Calculate initial score
    def get_score(mask):
        return sum(1 for ap_mask in three_aps if (mask & ap_mask) == ap_mask)
        
    current_score = get_score(current_mask)
    
    # Helper to get set elements
    def get_elements(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    # Incremental score update for swap (u -> v)
    def compute_swap_score(mask, u, v, current_score):
        u_bit = 1 << (u-1)
        v_bit = 1 << (v-1)
        new_mask = (mask & ~u_bit) | v_bit
        
        lost = 0
        for ap_mask in three_aps_by_v[u]:
            if (ap_mask & u_bit) and (ap_mask & ~u_bit & mask) == (ap_mask & ~u_bit):
                lost += 1
        
        gained = 0
        for ap_mask in three_aps_by_v[v]:
            if (ap_mask & v_bit) and (ap_mask & ~v_bit & new_mask) == (ap_mask & ~v_bit):
                gained += 1
        
        return current_score - lost + gained, new_mask
    
    # 3. Local Search with multiple passes
    while time.time() - start_time < DEADLINE:
        improved = False
        elements = get_elements(current_mask)
        non_elements = [i for i in range(1, N+1) if not (current_mask >> (i-1)) & 1]
        
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        for u in elements:
            if time.time() - start_time > DEADLINE:
                break
            u_bit = 1 << (u-1)
            for v in non_elements:
                if time.time() - start_time > DEADLINE:
                    break
                v_bit = 1 << (v-1)
                new_mask = (current_mask & ~u_bit) | v_bit
                
                valid = True
                for ap_mask in four_aps_by_v[v]:
                    if (new_mask & ap_mask) == ap_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                    
                new_score, _ = compute_swap_score(current_mask, u, v, current_score)
                
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # Try random kicks
            elements = get_elements(current_mask)
            non_elements = [i for i in range(1, N+1) if not (current_mask >> (i-1)) & 1]
            if elements and non_elements:
                u = random.choice(elements)
                v = random.choice(non_elements)
                u_bit = 1 << (u-1)
                v_bit = 1 << (v-1)
                new_mask = (current_mask & ~u_bit) | v_bit
                valid = True
                for ap_mask in four_aps_by_v[v]:
                    if (new_mask & ap_mask) == ap_mask:
                        valid = False
                        break
                if valid:
                    current_mask = new_mask
                    current_score = get_score(current_mask)
            else:
                break

    return get_elements(current_mask)
