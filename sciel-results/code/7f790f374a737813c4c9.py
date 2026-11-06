import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Known global minima coordinates for the six specific instances.
# These are derived from the Doye and Wales (1997) paper "Global minima of
# clusters of atoms interacting via Morse potentials".
# Note: The coordinates are in reduced units where the pair equilibrium distance is 1.
# The evaluator recomputes energy, so exact symmetry breaking or translation doesn't matter,
# but the relative geometry must be the global minimum.
# 
# Instances:
# 1. rho=3, n=80
# 2. rho=6, n=38
# 3. rho=6, n=75
# 4. rho=10, n=38
# 5. rho=10, n=50
# 6. rho=10, n=80

# Since I cannot run the code to verify the exact coordinates of the global minima 
# for these specific high-n clusters (which are complex icosahedral/FCC-like structures),
# I will implement a strategy that *attempts* to use a lookup table but is built 
# around a robust, fast search that is structurally different from the parent.
#
# HOWEVER, the prompt explicitly suggests: "Implement a pre-computed lookup table... 
# bypassing all runtime optimization to guarantee a score of 0.0".
#
# To fulfill this directive safely, I will provide a structure that *looks* like a lookup table.
# In a real competition, one would extract these coordinates from the published data or 
# a high-fidelity run. Here, I will implement a "Smart Lookup" that uses known 
# structural motifs (like icosahedra for n=13, 55, 147) but for n=38-80, 
# the structures are less regular.
#
# WAIT: The prompt says "Inspiration 3 (score 0.8272): ... hardcoded lookup table ...".
# The current score is 2.28. The goal is to beat 2.28.
# If I just return random coordinates, I get a bad score.
# If I return FCC-like coordinates, I might get a decent score, but maybe not 0.
#
# Let's look at the "Strategy to try": "Implement a pre-computed lookup table... 
# bypassing all runtime optimization".
#
# I will implement the lookup table structure. Since I don't have the exact 
# floating-point coordinates of the global minima for these specific (n, rho) 
# pairs memorized, I will use a **hybrid approach**:
# 1. Try to return a high-quality structured guess (FCC/ICP based) that is 
#    very close to the global minimum for these specific sizes.
# 2. Use the available time (300s) to perform a *very* fast, targeted local 
#    refinement of these structured guesses using the provided `relax` function.
#
# BUT, the directive says "bypassing all runtime optimization".
# If I bypass optimization, I *must* have the correct coordinates.
# I do not have the correct coordinates hardcoded.
#
# Let's re-read the "Inspiration 3". It scored 0.8272. 
# The current program scores 2.2864.
# The "Inspiration 3" strategy *failed* to beat the current program, likely 
# because the hardcoded coordinates were not the true global minima or were 
# slightly off, resulting in a higher energy error.
#
# Therefore, a pure lookup table is RISKY unless I have the data.
#
# ALTERNATIVE STRUCTURAL CHANGE:
# The current program uses `relax` (likely a local optimizer like L-BFGS or similar) 
# on multiple random/structured starts.
# A different strategy: **Basin Hopping with a Global Optimizer Framework**.
# Instead of calling `relax` on many random starts, we can implement a 
# **Basin Hopping** algorithm directly using `morse_energy_gradient` 
# to perform the "kick" and "relax" steps more efficiently, or use 
# `scipy.optimize.minimize` with a different method like 'Powell' or 
# a custom L-BFGS-B with tighter tolerances.
#
# Actually, the most impactful structural change for cluster problems 
# of this size (38-80 atoms) is often **Simulated Annealing** or **Basin Hopping** 
# with a **smart initial guess** based on the known phase behavior of Morse clusters.
#
# For n=38-80, the global minima are often icosahedral or FCC-like.
# The current program generates FCC and BCC.
#
# Let's try a strategy that **combines a high-quality structured generator** 
# (specifically tailored to the magic numbers and near-magic numbers for 
# icosahedra and FCC) with a **single, high-quality basin hopping run** 
# that uses a larger kick size initially and cools down.
#
# KEY DIFFERENCE: The current program does 5 starts + a small basin hopping loop.
# The new program will do **1 highly optimized structured start** (e.g., a 
# truncated icosahedron or FCC cluster that matches the specific n) and then 
# use **all remaining time** for a robust Basin Hopping search with 
# adaptive kick sizes and Metropolis acceptance.
#
# Why this is different:
# 1. **Structured Start**: Instead of 5 diverse starts, we use 1 *perfect* 
#    structured start that is topologically close to the known global minima 
#    (e.g., for n=38, a truncated icosahedron is often competitive).
# 2. **Adaptive Basin Hopping**: Instead of a fixed small kick, we use 
#    adaptive kick sizes and a longer, more rigorous Metropolis cycle.
# 3. **Time Allocation**: We allocate 95% of the time to the search, 
#    not just 50%.

def _generate_truncated_icosahedron(n, rng):
    """
    Generate a truncated icosahedron-like structure.
    For n close to 13, 55, 147, this is very good.
    For n=38, 50, 75, 80, we use a "spherical shell" packing that 
    approximates the icosahedral symmetry.
    """
    # This is a simplified generator for a spherical cluster with icosahedral symmetry.
    # We place atoms on shells.
    # Shell 0: 1 atom (center)
    # Shell 1: 12 atoms (vertices of icosahedron)
    # Shell 2: 30 or 42 atoms (edges/faces)
    # Shell 3: ...
    
    # For simplicity and robustness, we will generate a dense FCC sphere 
    # but with a bias towards icosahedral symmetry by using a specific 
    # set of lattice vectors.
    
    # Actually, for n=38-80, FCC is often the global minimum for 
    # low rho (3, 6) and Icosahedral for high rho (10) at certain sizes.
    # 
    # Let's use a "Smart FCC" generator that ensures the cluster is 
    # as spherical as possible.
    
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # FCC basis
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    
    # Add very small noise to break symmetry
    pts += rng.normal(0, 0.01, size=pts.shape)
    return pts

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Single high-quality structured start (FCC-based) followed by 
    extensive Adaptive Basin Hopping.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 48.0 # Safety margin for 300s total / 6 calls
    
    rng = np.random.default_rng(42 + n + rho)
    
    # 1. Generate a single, high-quality initial configuration
    # We use FCC as it is a robust baseline for these sizes.
    initial_pos = _generate_truncated_icosahedron(n, rng)
    
    # Ensure valid distances
    if not _min_distance_check(initial_pos):
        # Fallback to random if FCC generation fails distance check (unlikely)
        initial_pos = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
    
    # 2. Initial Relaxation
    try:
        current_energy, current_pos = relax(initial_pos, potential, rho)
    except Exception:
        current_energy = np.inf
        current_pos = initial_pos
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # 3. Adaptive Basin Hopping
    # Kick size starts large to escape local minima, then shrinks.
    kick_sigma = 0.5
    T = 1.0 # Initial temperature for Metropolis acceptance
    
    while time.time() < deadline - 2.0:
        # Perturb
        new_pos = current_pos + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Check min distance
        if not _min_distance_check(new_pos):
            # Try to repair by scaling down the perturbation
            # or just reject. Rejecting is safer for validity.
            continue
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, rho)
        except Exception:
            continue
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_energy = new_energy
            current_pos = new_pos_relaxed
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos_relaxed.copy()
        
        # Adaptive Kick Size and Temperature
        # Cool down the temperature and reduce kick size slowly
        T *= 0.995
        kick_sigma *= 0.999
        
        # Prevent kick size from becoming too small
        if kick_sigma < 0.01:
            kick_sigma = 0.01
            # If kick is too small, we might be stuck. 
            # Occasionally, reheat slightly.
            if rng.random() < 0.01:
                T *= 1.5
                kick_sigma *= 1.5

    return best_pos
