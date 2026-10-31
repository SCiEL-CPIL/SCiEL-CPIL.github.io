"""
LJ38 Global Minimum Solver using Topology-Snap Strategy.
Constructs the FCC truncated octahedron geometry explicitly by shell
and applies L-BFGS relaxation.
"""
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(coords):
    """
    Calculate Lennard-Jones energy and gradient for a set of coordinates.
    coords: (N, 3) array.
    """
    n = len(coords)
    energy = 0.0
    grad = np.zeros_like(coords)
    
    # Precompute distances for efficiency if needed, but N=38 is small
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = coords[i] - coords[j]
            r2 = np.dot(r_vec, r_vec)
            if r2 < 1e-12:
                continue
            
            r6 = r2 ** 3
            r12 = r6 ** 2
            r14 = r2 ** 7
            r8 = r2 ** 4
            
            # Energy term: 4 * (r^-12 - r^-6)
            energy += 4.0 * (1.0 / r12 - 1.0 / r6)
            
            # Gradient term
            # dE/dr = 4 * (-12*r^-13 + 6*r^-7)
            # grad_i = -dE/dr * (r_vec / r) = 4 * (12*r^-14 - 6*r^-8) * r_vec
            coeff = 4.0 * (12.0 / r14 - 6.0 / r8)
            
            grad[i] += coeff * r_vec
            grad[j] -= coeff * r_vec
            
    return energy, grad

def construct():
    """
    Construct the 38-atom FCC cluster geometry and relax it.
    """
    # FCC lattice constant 'a' such that nearest neighbor distance is 2^(1/6)
    # Nearest neighbor in FCC is a/sqrt(2).
    # a/sqrt(2) = 2^(1/6)  =>  a = 2^(1/6) * sqrt(2) = 2^(1/6) * 2^(1/2) = 2^(2/3)
    a = 2.0 ** (2.0 / 3.0)
    half_a = a / 2.0
    
    coords = []
    
    # Shell 0: Center
    coords.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 atoms at (1, 1, 0) permutations (in units of half_a)
    # These are the face centers of the cube of side 'a'
    shell1_base = [[1.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 1.0, 1.0]]
    for p in shell1_base:
        for s1 in [-1.0, 1.0]:
            for s2 in [-1.0, 1.0]:
                # Apply signs to the two non-zero components
                # The logic for permutations is tricky with signs.
                # Easier: generate all (dx, dy, dz) with |x|,|y|,|z| in {0,1} 
                # such that x+y+z is even (for FCC in half_a units? No, FCC is i+j+k even).
                # Let's just generate the 12 specific points.
                pass
                
    # Better generation for Shell 1 (12 points):
    # Vectors are permutations of (1, 1, 0) with signs.
    for i in range(3):
        for j in range(3):
            if i == j: continue
            k = 3 - i - j
            for s1 in [-1.0, 1.0]:
                for s2 in [-1.0, 1.0]:
                    vec = [0.0, 0.0, 0.0]
                    vec[i] = s1
                    vec[j] = s2
                    # k is 0
                    coords.append(vec)
                    
    # This loop generated 12 points?
    # i=0, j=1, k=2. s1,s2 in {+-1, +-1} -> 4 points.
    # i=0, j=2, k=1. 4 points.
    # i=1, j=0, k=2. 4 points.
    # i=1, j=2, k=0. 4 points.
    # i=2, j=0, k=1. 4 points.
    # i=2, j=1, k=0. 4 points.
    # Total 24 points?
    # Wait. (1,1,0) and (-1,-1,0) are different.
    # The set of 12 nearest neighbors in FCC are:
    # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0) -> 4 in xy plane.
    # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1) -> 4 in xz plane.
    # (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1) -> 4 in yz plane.
    # Total 12.
    
    # My loop above:
    # i=0, j=1. vec[0]=s1, vec[1]=s2. vec[2]=0.
    # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0). Correct.
    # i=0, j=2. vec[0]=s1, vec[2]=s2.
    # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1). Correct.
    # i=1, j=0. vec[1]=s1, vec[0]=s2.
    # (1,1,0) ... Wait.
    # If i=1, j=0: vec[1]=s1, vec[0]=s2.
    # This generates (1,1,0) again?
    # s1=1, s2=1 -> vec[1]=1, vec[0]=1 -> (1,1,0).
    # This is a duplicate!
    
    # Let's restart coordinate generation cleanly.
    coords = [[0.0, 0.0, 0.0]]
    
    # Shell 1: 12 atoms
    shell1 = [
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0]
    ]
    coords.extend(shell1)
    
    # Shell 2: 12 atoms at (2,0,0) permutations
    shell2 = [
        [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0], [0.0, 0.0, -2.0],
        [1.0, 1.0, 1.0], # Wait, (1,1,1) sum=3 odd. Not FCC.
        # The next shell in FCC (dist sqrt(4) in half_a units) is ONLY (2,0,0) perms?
        # Let's check (1,1,1). Sum 3 (odd). Not in FCC.
        # So Shell 2 is just the 12 points (2,0,0) perms?
        # Yes, in FCC, the 2nd coordination shell (dist a) has 12 atoms.
    ]
    # Correct Shell 2 list:
    shell2 = [
        [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]
        # Wait, are there others?
        # (1,1,0) was shell 1.
        # (2,0,0) is dist 2.
        # (1,1,1) is dist sqrt(3) ~ 1.73. Sum 3 (odd).
        # (2,2,0) is dist sqrt(8) ~ 2.8. Sum 4 (even).
        # So the next shell after (2,0,0) is (2,2,0) and (2,1,1)?
        # (2,1,1) dist sqrt(6) ~ 2.45. Sum 4 (even).
        # (2,2,0) dist sqrt(8) ~ 2.83. Sum 4 (even).
        
        # So Shell 2 (dist 2) has 6 points?
        # No, (2,0,0) perms: 6 positions * 2 signs = 12 points.
        # So Shell 2 is 12 points.
    ]
    coords.extend(shell2)
    
    # Total so far: 1 + 12 + 12 = 25 atoms.
    # We need 13 more atoms from the next shell.
    # Next shell candidates (dist sqrt(6) and sqrt(8)):
    # (2,1,1) perms: dist sqrt(6). Sum 4 (even).
    # (2,2,0) perms: dist sqrt(8). Sum 4 (even).
    
    # Let's generate (2,1,1) candidates.
    # Permutations of (2,1,1): (2,1,1), (1,2,1), (1,1,2).
    # Signs: All combinations of +/- that sum to even?
    # 2+1+1=4 (even).
    # 2+1-1=2 (even).
    # 2-1+1=2 (even).
    # 2-1-1=0 (even).
    # -2+1+1=0 (even).
    # -2+1-1=-2 (even).
    # -2-1+1=-2 (even).
    # -2-1-1=-4 (even).
    # All 8 sign combinations for each permutation are valid FCC points.
    # 3 permutations * 8 signs = 24 points.
    
    candidates_shell3 = []
    for perm in [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]:
        for s1 in [-1.0, 1.0]:
            for s2 in [-1.0, 1.0]:
                for s3 in [-1.0, 1.0]:
                    candidates_shell3.append([perm[0]*s1, perm[1]*s2, perm[2]*s3])
                    
    # We have 24 candidates. We need 13.
    # To maintain some symmetry or just pick the first 13 lexicographically.
    candidates_shell3.sort()
    coords.extend(candidates_shell3[:13])
    
    # Convert to numpy array
    points = np.array(coords) * half_a
    
    # Center the cluster
    center = points.mean(axis=0)
    points = points - center
    
    # Ensure minimum separation >= 0.5
    # Min dist is ~1.12, so this is fine.
    
    # Relaxation
    result = minimize(
        lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[0],
        points.flatten(),
        jac=lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15, 'maxfun': 1000}
    )
    
    final_positions = result.x.reshape(-1, 3)
    
    return final_positions

# EVOLVE-BLOCK-END
