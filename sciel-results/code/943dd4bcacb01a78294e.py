import time
import random
import itertools

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute masks by size for fast alpha calculation
MASKS_BY_SIZE = [[] for _ in range(N_POINTS + 1)]
for m in range(N_MASKS):
    MASKS_BY_SIZE[POP_COUNT[m]].append(m)

# Generate all 2D affine subspaces (planes) in AG(4,2)
# Points are 0..15 (vectors in F2^4)
# A 2D affine subspace is defined by a base point and a 2D linear subspace.
# There are 35 such planes.

def get_planes():
    planes = []
    seen = set()
    # All 2D linear subspaces of F2^4
    # A 2D linear subspace is spanned by two independent vectors u, v.
    # There are (2^4-1)(2^4-2) / ( (2^2-1)(2^2-2) ) = 15*14 / 6 = 35? No.
    # Number of 2D subspaces in F2^4 is 35.
    # Let's generate them explicitly.
    
    linear_subspaces = []
    for u in range(1, 16):
        for v in range(u + 1, 16):
            # Check independence: u and v are independent if u != v and u != 0 and v != 0
            # In F2, independent means v is not a multiple of u. Since only multiple is 1, u != v is enough.
            # But we need to avoid duplicates.
            # The subspace is {0, u, v, u^v}
            sub = {0, u, v, u ^ v}
            # Canonical representation: sorted tuple
            key = tuple(sorted(sub))
            if key not in seen:
                seen.add(key)
                linear_subspaces.append(sub)
                
    # For each linear subspace L, there are 16/4 = 4 cosets (affine planes).
    # Cosets are L + x for x in F2^4.
    # We only need to generate each plane once.
    all_planes = set()
    for L in linear_subspaces:
        # L is a set of 4 points
        # The 4 cosets are L, L^1, L^2, ... 
        # Actually, just iterate x from 0 to 15.
        # Plane is {p ^ x for p in L}
        for x in range(16):
            plane = {p ^ x for p in L}
            # Canonical representation
            key = tuple(sorted(plane))
            if key not in all_planes:
                all_planes.add(key)
                
    # There should be 35 planes? 
    # Wait. Number of 2D affine subspaces in AG(4,2).
    # Formula: (2^4-1)(2^4-2)(2^4-4) / ( (2^2-1)(2^2-2)(2^2-4) ) ? No.
    # Number of k-planes in AG(n,q) is product_{i=0}^{k-1} (q^{n-i}-1)/(q^k-q^i) * q^{n-k} ?
    # For n=4, k=2, q=2:
    # # of 2D linear subspaces = 35.
    # Each has 4 cosets.
    # Total planes = 35 * 4 / 4? No, each plane is a coset of exactly one linear subspace.
    # So total planes = 35 * 4 = 140?
    # Let's check.
    # Total points 16.
    # Each plane has 4 points.
    # Total incidences = 140 * 4 = 560.
    # Each point is in how many planes?
    # Number of planes through a point = Number of 2D linear subspaces in F2^3 (directions) = 7.
    # 16 points * 7 planes/point = 112 incidences?
    # Discrepancy.
    # Ah, the linear subspaces are in the vector space.
    # The number of 2D linear subspaces in F2^4 is 35.
    # Each point is the origin of exactly one copy of each linear subspace? No.
    # For a fixed point p, the planes containing p are in bijection with the 2D linear subspaces of F2^4.
    # So each point is in 35 planes.
    # Total incidences = 16 * 35 = 560.
    # Total planes = 560 / 4 = 140.
    # So there are 140 planes.
    
    return [list(p) for p in all_planes]

PLANES = get_planes()
NUM_PLANES = len(PLANES)

# For each plane, get the 4 triples
# A plane has 4 points {a,b,c,d}.
# Triples are {a,b,c}, {a,b,d}, {a,c,d}, {b,c,d}.
PLANE_TRIPLES = []
for p in PLANES:
    a, b, c, d = p
    t1 = (1 << a) | (1 << b) | (1 << c)
    t2 = (1 << a) | (1 << b) | (1 << d)
    t3 = (1 << a) | (1 << c) | (1 << d)
    t4 = (1 << b) | (1 << c) | (1 << d)
    PLANE_TRIPLES.append([t1, t2, t3, t4])

# Precompute conflicts between planes
# conflict[i][j][k][l] = True if triple k of plane i and triple l of plane j share >= 2 points.
# We can store this as a bitmask or a list of allowed l for each k.
# allowed[i][j][k] = bitmask of l in 0..3 such that T_i[k] and T_j[l] are compatible.
COMPATIBILITY = [[[0]*4 for _ in range(NUM_PLANES)] for __ in range(NUM_PLANES)]

for i in range(NUM_PLANES):
    for j in range(i + 1, NUM_PLANES):
        for k in range(4):
            mask_i = PLANE_TRIPLES[i][k]
            compat_mask = 0
            for l in range(4):
                mask_j = PLANE_TRIPLES[j][l]
                if (mask_i & mask_j).bit_count() < 2:
                    compat_mask |= (1 << l)
            COMPATIBILITY[i][j][k] = compat_mask
            COMPATIBILITY[j][i][l] = 0 # Placeholder, will fill symmetrically below? No, indices are k, l.
            # Wait, COMPATIBILITY[i][j][k] gives valid l's.
            # COMPATIBILITY[j][i][l] should give valid k's.
            # We can just compute it.

# Recompute symmetrically to be safe
COMPATIBILITY = [[[0]*4 for _ in range(NUM_PLANES)] for __ in range(NUM_PLANES)]
for i in range(NUM_PLANES):
    for j in range(i + 1, NUM_PLANES):
        for k in range(4):
            mask_i = PLANE_TRIPLES[i][k]
            valid_l = 0
            for l in range(4):
                mask_j = PLANE_TRIPLES[j][l]
                if (mask_i & mask_j).bit_count() < 2:
                    valid_l |= (1 << l)
            COMPATIBILITY[i][j][k] = valid_l
            # Symmetric part: for each valid l, k is valid for l
            for l in range(4):
                if valid_l & (1 << l):
                    COMPATIBILITY[j][i][l] |= (1 << k)

def construct():
    deadline = time.time() + 110.0
    _random = random.Random(42)
    
    # State: choice[i] is the index (0..3) of the triple chosen for plane i.
    choice = [_random.randrange(4) for _ in range(NUM_PLANES)]
    
    # Precompute blocked_count for current choice
    # blocked_count[m] = number of triples in system subset of m
    blocked_count = [0] * N_MASKS
    
    def rebuild_blocked():
        for m in range(N_MASKS):
            blocked_count[m] = 0
        for i in range(NUM_PLANES):
            t = PLANE_TRIPLES[i][choice[i]]
            # Update supersets
            comp = ((1 << N_POINTS) - 1) ^ t
            sub = comp
            while True:
                blocked_count[t | sub] += 1
                if sub == 0:
                    break
                sub = (sub - 1) & comp

    def update_blocked(remove_plane, remove_idx, add_plane, add_idx):
        # Remove old triple
        t_old = PLANE_TRIPLES[remove_plane][remove_idx]
        comp = ((1 << N_POINTS) - 1) ^ t_old
        sub = comp
        while True:
            blocked_count[t_old | sub] -= 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
        # Add new triple
        t_new = PLANE_TRIPLES[add_plane][add_idx]
        comp = ((1 << N_POINTS) - 1) ^ t_new
        sub = comp
        while True:
            blocked_count[t_new | sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp

    def get_alpha():
        # Find max size of independent set
        # An independent set is a mask with blocked_count[m] == 0
        for k in range(N_POINTS, 0, -1):
            for m in MASKS_BY_SIZE[k]:
                if blocked_count[m] == 0:
                    return k
        return 0

    def is_valid_assignment():
        # Check if current choice is linear
        # We can check conflicts on the fly or precomputed.
        # Since we only change one plane at a time, we can just check the changed plane against others.
        # But for initial repair, we need full check.
        for i in range(NUM_PLANES):
            for j in range(i + 1, NUM_PLANES):
                ki = choice[i]
                kj = choice[j]
                if not (COMPATIBILITY[i][j][ki] & (1 << kj)):
                    return False
        return True

    def count_conflicts():
        count = 0
        for i in range(NUM_PLANES):
            for j in range(i + 1, NUM_PLANES):
                ki = choice[i]
                kj = choice[j]
                if not (COMPATIBILITY[i][j][ki] & (1 << kj)):
                    count += 1
        return count

    # 1. Repair phase: Find a valid assignment
    # Start random
    while time.time() < deadline:
        if not is_valid_assignment():
            # Find a random conflict
            conflicts = []
            for i in range(NUM_PLANES):
                for j in range(i + 1, NUM_PLANES):
                    ki = choice[i]
                    kj = choice[j]
                    if not (COMPATIBILITY[i][j][ki] & (1 << kj)):
                        conflicts.append((i, j))
            if not conflicts:
                break
            i, j = _random.choice(conflicts)
            # Try to change choice[i] to resolve
            ki = choice[i]
            found = False
            for new_k in range(4):
                if new_k == ki:
                    continue
                # Check if new_k is compatible with all other planes
                ok = True
                for l in range(NUM_PLANES):
                    if l == i:
                        continue
                    kl = choice[l]
                    if not (COMPATIBILITY[i][l][new_k] & (1 << kl)):
                        ok = False
                        break
                if ok:
                    choice[i] = new_k
                    found = True
                    break
            if not found:
                # Try changing choice[j]
                kj = choice[j]
                for new_k in range(4):
                    if new_k == kj:
                        continue
                    ok = True
                    for l in range(NUM_PLANES):
                        if l == j:
                            continue
                        kl = choice[l]
                        if not (COMPATIBILITY[j][l][new_k] & (1 << kl)):
                            ok = False
                            break
                    if ok:
                        choice[j] = new_k
                        found = True
                        break
            if not found:
                # Random restart
                choice = [_random.randrange(4) for _ in range(NUM_PLANES)]
        else:
            break
            
    if time.time() > deadline - 10:
        # Fallback: return a trivial valid system if repair takes too long
        # Just pick one plane's triple? No, must be valid.
        # If we can't find a valid one, return empty? 
        # But empty has alpha 16.
        # We should have found one.
        pass

    rebuild_blocked()
    current_alpha = get_alpha()
    best_alpha = current_alpha
    best_choice = choice[:]
    
    # 2. Local Search
    while time.time() < deadline - 5:
        # Try to improve
        improved = False
        for i in range(NUM_PLANES):
            old_k = choice[i]
            for new_k in range(4):
                if new_k == old_k:
                    continue
                # Check compatibility
                ok = True
                for l in range(NUM_PLANES):
                    if l == i:
                        continue
                    kl = choice[l]
                    if not (COMPATIBILITY[i][l][new_k] & (1 << kl)):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Try the move
                choice[i] = new_k
                update_blocked(i, old_k, i, new_k)
                new_alpha = get_alpha()
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_choice = choice[:]
                    improved = True
                elif new_alpha <= current_alpha:
                    # Accept to explore
                    current_alpha = new_alpha
                    improved = True
                else:
                    # Revert
                    choice[i] = old_k
                    update_blocked(i, new_k, i, old_k)
        
        if not improved:
            # Random kick
            i = _random.randrange(NUM_PLANES)
            old_k = choice[i]
            new_k = _random.randrange(4)
            if new_k != old_k:
                # Check compat
                ok = True
                for l in range(NUM_PLANES):
                    if l == i: continue
                    kl = choice[l]
                    if not (COMPATIBILITY[i][l][new_k] & (1 << kl)):
                        ok = False
                        break
                if ok:
                    choice[i] = new_k
                    update_blocked(i, old_k, i, new_k)
                    current_alpha = get_alpha()
                else:
                    # Try to find a valid kick
                    for attempt in range(10):
                        new_k = _random.randrange(4)
                        if new_k == old_k: continue
                        ok = True
                        for l in range(NUM_PLANES):
                            if l == i: continue
                            kl = choice[l]
                            if not (COMPATIBILITY[i][l][new_k] & (1 << kl)):
                                ok = False
                                break
                        if ok:
                            choice[i] = new_k
                            update_blocked(i, old_k, i, new_k)
                            current_alpha = get_alpha()
                            break

    # Convert best_choice to result
    result = []
    for i in range(NUM_PLANES):
        t_mask = PLANE_TRIPLES[i][best_choice[i]]
        pts = []
        for p in range(N_POINTS):
            if t_mask & (1 << p):
                pts.append(p)
        result.append(pts)
        
    return result

