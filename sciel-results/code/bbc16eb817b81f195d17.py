"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a broadband antireflection coating using a pre-optimized stack.
    Uses a combination of quarter-wave and half-wave layers at the center wavelength
    to minimize mean reflectance over 400-800 nm.
    
    The design uses a 4-layer stack: H-L-L-H (or similar symmetric structure)
    with thicknesses optimized for broadband performance.
    """
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    nL = materials["L"]
    nH = materials["H"]
    n0 = ambient_index
    ns = substrate_index
    
    centre = 0.5 * (band_nm[0] + band_nm[1])  # 600 nm
    
    # Use a 5-layer symmetric stack: L-H-L-H-L
    # This is a common broadband AR coating design
    # Thicknesses are in units of quarter-wavelength at center wavelength
    # For broadband performance, use slightly different thicknesses
    
    # Let's try a well-known design: 
    # Layer 1 (L): ~0.25 lambda_c
    # Layer 2 (H): ~0.25 lambda_c  
    # Layer 3 (L): ~0.5 lambda_c (half-wave)
    # Layer 4 (H): ~0.25 lambda_c
    # Layer 5 (L): ~0.25 lambda_c
    
    # Physical thickness = (optical thickness) / index
    # Quarter wave at center: lambda_c / (4 * n)
    # Half wave at center: lambda_c / (2 * n)
    
    qL = centre / (4.0 * nL)  # quarter-wave L layer thickness
    qH = centre / (4.0 * nH)  # quarter-wave H layer thickness
    hL = centre / (2.0 * nL)  # half-wave L layer thickness
    
    # Try a 5-layer symmetric design
    # L-H-L-H-L with thicknesses [qL, qH, hL, qH, qL]
    materials_list = ["L", "H", "L", "H", "L"]
    thicknesses = [qL, qH, hL, qH, qL]
    
    # Check constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down proportionally
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    # Ensure each thickness is within [0, max_layer_nm]
    thicknesses = [max(0.0, min(t, max_layer_nm)) for t in thicknesses]
    
    # Final constraint check
    if len(materials_list) > max_layers:
        # Truncate (shouldn't happen with 5 layers)
        materials_list = materials_list[:max_layers]
        thicknesses = thicknesses[:max_layers]
    
    return materials_list, thicknesses
# EVOLVE-BLOCK-END
