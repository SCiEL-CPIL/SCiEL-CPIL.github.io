"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Center inputs: range [0.5, 2], center at 1.25
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25

    p = params

    # 10 terms: constant, linear, radial quadratic, and cross quadratic
    # 0: constant
    # 1: cx
    # 2: cy
    # 3: cz
    # 4: isotropic radial quadratic (cx^2+cy^2+cz^2)
    # 5-9: cross quadratic terms (cx*cy, cx*cz, cy*cz) plus two more
    # Actually with 10 params: const(1) + linear(3) + radial quad(1) + cross quad(3) = 8
    # Need 2 more: use cx^2 and cy^2 separately? Let me think...
    # 
    # Better: const, cx, cy, cz, cx^2+cy^2+cz^2 (radial), cx*cy, cx*cz, cy*cz,
    # and add x*y*z interaction and a pure cubic-like term?
    #
    # Let me use: const, 3 linear, 1 radial quadratic, 3 cross quadratic = 8 terms
    # Remaining 2: add cx*cy*cz and (cx^2 - cy^2) for anisotropic quadratic
    #
    # Actually simpler: use all 10 for a richer but still well-conditioned basis:
    # 0: 1
    # 1: cx
    # 2: cy  
    # 3: cz
    # 4: cx^2 + cy^2 + cz^2  (isotropic quadratic)
    # 5: cx * cy
    # 6: cx * cz
    # 7: cy * cz
    # 8: cx * cy * cz  (trilinear interaction)
    # 9: cx^2 - cy^2  (anisotropic quadratic component)

    r2 = cx * cx + cy * cy + cz * cz
    cross1 = cx * cy
    cross2 = cx * cz
    cross3 = cy * cz
    tri = cx * cy * cz
    aniso = cx * cx - cy * cy

    return (
        p[0]
        + p[1] * cx
        + p[2] * cy
        + p[3] * cz
        + p[4] * r2
        + p[5] * cross1
        + p[6] * cross2
        + p[7] * cross3
        + p[8] * tri
        + p[9] * aniso
    )
# EVOLVE-BLOCK-END
