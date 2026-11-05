"""Broadband antireflection coating design via Genetic Algorithm with vectorized evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    thicknesses: array of physical thicknesses in nm
    wls: array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        R = r * r
        return float(R)
    
    # Initialize total characteristic matrix M = I
    # Shape: (len(wls), 2, 2)
    M = np.zeros((len(wls), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    # Pre-compute wavelength-dependent terms for efficiency? 
    # No, just loop layers. Vectorize over wavelengths.
    for i in range(n_layers):
        if i % 2 == 0:
            n = n_H
        else:
            n = n_L
        d = thicknesses[i]
        
        # beta = 2 * pi * n * d / wl
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix L
        # L11 = cos_b
        # L12 = -1j * sin_b / n
        # L21 = 1j * n * sin_b
        # L22 = cos_b
        
        # M_new = M @ L
        # M[:, 0, 0] * L11 + M[:, 0, 1] * L21
        new_M11 = M[:, 0, 0] * cos_b + M[:, 0, 1] * (1j * n * sin_b)
        # M[:, 0, 0] * L12 + M[:, 0, 1] * L22
        new_M12 = M[:, 0, 0] * (-1j * sin_b / n) + M[:, 0, 1] * cos_b
        # M[:, 1, 0] * L11 + M[:, 1, 1] * L21
        new_M21 = M[:, 1, 0] * cos_b + M[:, 1, 1] * (1j * n * sin_b)
        # M[:, 1, 0] * L12 + M[:, 1, 1] * L22
        new_M22 = M[:, 1, 0] * (-1j * sin_b / n) + M[:, 1, 1] * cos_b
        
        M[:, 0, 0] = new_M11
        M[:, 0, 1] = new_M12
        M[:, 1, 0] = new_M21
        M[:, 1, 1] = new_M22
    
    # Admittance Y
    # Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M[:, 0, 0] * n_substrate + M[:, 0, 1]) / (M[:, 1, 0] * n_substrate + M[:, 1, 1])
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r)**2
    
    return float(np.mean(R))

def _penalized_objective(x, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm, max_layers):
    """
    Objective function for GA. Returns mean reflectance + penalties for violations.
    x: array of length max_layers, where x[i] is thickness of layer i (0 means removed).
    """
    # Extract active layers (thickness > 0)
    # We enforce a fixed material sequence H, L, H, L... in the GA representation
    # to simplify the search space structure, but we allow "removal" via 0 thickness.
    # Actually, removing a layer changes the material sequence for subsequent layers.
    # To handle this correctly, we should interpret x as a sequence of 20 slots.
    # If slot i is 0, it is empty.
    # The material of slot i is determined by its index i (even=H, odd=L) REGARDLESS of previous empties?
    # No, physical stack order matters.
    # If I define the design as a fixed sequence of 20 layers where some can be 0,
    # the material index i corresponds to the i-th layer in the stack, not the i-th non-zero layer.
    # This allows "holes" which are physically just air gaps? No, 0 thickness means the interface is merged.
    # If layer i is 0, it effectively disappears. The next layer i+1 becomes adjacent to i-1.
    # The material of layer i+1 is still determined by index i+1?
    # In the problem statement: "list of 'L'/'H' labels".
    # If I output [H, 0, L], the materials are H, L.
    # My GA representation: x = [t0, t1, ..., t19].
    # Materials are fixed by index: 0->H, 1->L, 2->H...
    # If t1=0, we have H, then L (index 2).
    # This is a valid interpretation.
    
    active_indices = [i for i in range(len(x)) if x[i] > 1e-6]
    if not active_indices:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r*r) * 100.0

    active_thick = x[active_indices]
    
    # Penalties
    penalty = 0.0
    if len(active_indices) > max_layers:
        penalty += 100.0
    if np.sum(active_thick) > max_total_nm:
        penalty += 10.0 * (np.sum(active_thick) - max_total_nm)
    if np.any(active_thick > max_layer_nm):
        penalty += 10.0 * (np.max(active_thick) - max_layer_nm)
        # Also clamp for calculation to avoid huge numbers, but penalty handles it
        
    # Calculate reflectance
    # Note: _vectorized_reflectance expects thicknesses and determines materials by index parity.
    # It assumes the input list corresponds to the physical stack order.
    # My `active_thick` is in order.
    # But `_vectorized_reflectance` uses `i % 2` to determine material.
    # If I skip an index, the parity in the function might not match the intended material?
    # Let's check `_vectorized_reflectance`:
    # `for i in range(n_layers): if i % 2 == 0: n = n_H else: n = n_L`
    # This assumes the 0-th layer is H, 1-st is L, etc.
    # If my active stack is [H(idx0), L(idx2)], the function will see 2 layers.
    # Layer 0 (idx 0) -> H. Correct.
    # Layer 1 (idx 2) -> L. Correct.
    # So the parity in the function matches the index parity of the original slots?
    # Yes, because I pass the active thicknesses in order, and the function assigns H/L based on position in the *passed* list.
    # Wait.
    # If I have slots 0(H), 1(L), 2(H).
    # If I remove slot 1, active are 0(H), 2(H).
    # Passed list: [t0, t2].
    # Function sees:
    # i=0: H. Correct.
    # i=1: L. **INCORRECT**. It should be H (from slot 2).
    
    # So the current `_vectorized_reflectance` is NOT compatible with "skipping" layers if material depends on original index.
    # Solution: Pass the material sequence explicitly or adjust the function.
    # Let's modify the approach:
    # Instead of skipping, let's just use a fixed number of layers?
    # Or, better: The problem allows *any* sequence of L/H.
    # My GA representation:
    # We can encode the design as a sequence of 20 (material, thickness) pairs?
    # That doubles the dimensionality.
    # Alternative:
    # Since we can choose materials freely, maybe we should optimize a fixed number of layers N?
    # But N is a variable.
    
    # Let's stick to the "fixed 20 slots, 0 means empty" but fix the material assignment.
    # We need to pass the specific material for each active layer to the reflectance calculator.
    
    # Let's write a specialized reflectance function that takes a list of (n, d) pairs.
    
    pass

def _reflectance_with_materials(pairs, n_ambient, n_substrate, wls):
    """
    pairs: list of (n, d) tuples.
    """
    if not pairs:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r*r)

    M = np.zeros((len(wls), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for n, d in pairs:
        if d < 1e-6:
            continue
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M[:, 0, 0] * L11 + M[:, 0, 1] * L21
        new_M12 = M[:, 0, 0] * L12 + M[:, 0, 1] * L22
        new_M21 = M[:, 1, 0] * L11 + M[:, 1, 1] * L21
        new_M22 = M[:, 1, 0] * L12 + M[:, 1, 1] * L22
        
        M[:, 0, 0] = new_M11
        M[:, 0, 1] = new_M12
        M[:, 1, 0] = new_M21
        M[:, 1, 1] = new_M22
        
    Y = (M[:, 0, 0] * n_substrate + M[:, 0, 1]) / (M[:, 1, 0] * n_substrate + M[:, 1, 1])
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r)**2
    return float(np.mean(R))

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    wls = np.linspace(wl_min, wl_max, 401)
    
    # GA Parameters
    pop_size = 200
    generations = 50
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_count = 10
    
    # Representation:
    # x: array of size max_layers * 2
    # x[0] = thickness of layer 0
    # x[1] = material of layer 0 (0 for L, 1 for H)
    # ...
    # This allows any material sequence.
    # Dimensionality: 40.
    
    def decode(x):
        """
        Decode GA chromosome to (materials_list, thicknesses_list)
        """
        mats = []
        thicks = []
        for i in range(max_layers):
            t = x[i*2]
            m = x[i*2+1]
            if t < 1e-6:
                continue
            # Clamp thickness
            t = np.clip(t, 0, max_layer_nm)
            # Material
            if m > 0.5:
                mats.append('H')
            else:
                mats.append('L')
            thicks.append(float(t))
        return mats, thicks

    def evaluate_chromosome(x):
        mats, thicks = decode(x)
        if not thicks:
            r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
            return float(r*r) * 100.0
        
        # Check constraints
        penalty = 0.0
        if len(thicks) > max_layers:
            penalty += 100.0
        total_t = sum(thicks)
        if total_t > max_total_nm:
            penalty += 10.0 * (total_t - max_total_nm)
        
        # Build pairs
        pairs = []
        for m, t in zip(mats, thicks):
            n = n_H if m == 'H' else n_L
            pairs.append((n, t))
            
        R_frac = _reflectance_with_materials(pairs, n_ambient, n_substrate, wls)
        return R_frac * 100.0 + penalty

    # Initialize Population
    np.random.seed(42)
    population = np.random.rand(pop_size, max_layers * 2)
    # Initialize thicknesses: random in [0, max_layer_nm]
    # Initialize materials: random 0 or 1
    for i in range(pop_size):
        # Thicknesses
        population[i, 0::2] = np.random.rand(max_layers) * max_layer_nm
        # Materials
        population[i, 1::2] = np.random.randint(0, 2, max_layers).astype(float)
        
        # Force some layers to be 0 to encourage sparsity (fewer layers)
        # Randomly zero out 50% of layers
        zero_mask = np.random.rand(max_layers) < 0.5
        population[i, zero_mask * 2] = 0.0

    # Evaluate initial population
    scores = np.array([evaluate_chromosome(ind) for ind in population])
    
    best_score = np.min(scores)
    best_chrom = population[np.argmin(scores)]
    
    # GA Loop
    gen = 0
    while gen < generations:
        if time.time() > deadline - 5.0:
            break
            
        # Sort population
        order = np.argsort(scores)
        population = population[order]
        scores = scores[order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_chrom = population[0].copy()
            
        # Elitism
        new_population = []
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Create offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 2.0:
                break
                
            # Tournament selection
            def select():
                idx1 = np.random.randint(0, pop_size)
                idx2 = np.random.randint(0, pop_size)
                if scores[idx1] < scores[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            p1 = select()
            p2 = select()
            
            if np.random.rand() < crossover_rate:
                # Uniform crossover
                mask = np.random.rand(len(p1)) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            if np.random.rand() < mutation_rate:
                # Mutate a random gene
                gene_idx = np.random.randint(0, len(child))
                if gene_idx % 2 == 0:
                    # Thickness mutation
                    # Gaussian perturbation
                    perturbation = np.random.normal(0, 20.0)
                    child[gene_idx] += perturbation
                    child[gene_idx] = np.clip(child[gene_idx], 0, max_layer_nm)
                    # Sometimes reset to 0
                    if np.random.rand() < 0.1:
                        child[gene_idx] = 0.0
                else:
                    # Material mutation
                    child[gene_idx] = 1.0 - child[gene_idx]
            
            # Occasionally add a new random individual to maintain diversity
            if np.random.rand() < 0.05:
                child = np.random.rand(max_layers * 2)
                child[0::2] = np.random.rand(max_layers) * max_layer_nm
                child[1::2] = np.random.randint(0, 2, max_layers).astype(float)
                zero_mask = np.random.rand(max_layers) < 0.5
                child[zero_mask * 2] = 0.0
                
            new_population.append(child)
            
        population = np.array(new_population)
        scores = np.array([evaluate_chromosome(ind) for ind in population])
        gen += 1

    # Final Result
    mats, thicks = decode(best_chrom)
    
    # Fallback if empty or invalid
    if not thicks:
        # Simple quarter wave L layer
        wl_center = 0.5 * (wl_min + wl_max)
        qw_L = wl_center / (4.0 * n_L)
        mats = ["L"]
        thicks = [qw_L]
        
    # Ensure validity
    if sum(thicks) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(thicks)
        thicks = [t * scale for t in thicks]
        
    if len(thicks) > max_layers:
        mats = mats[:max_layers]
        thicks = thicks[:max_layers]

    return mats, thicks
# EVOLVE-BLOCK-END
