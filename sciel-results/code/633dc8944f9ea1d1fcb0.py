"""Photonic-crystal unit cell: dielectric ring (annulus) for a higher-order TM gap.

The annulus introduces a broad spectrum of dielectric Fourier components, which
is what opens a gap between higher-order (4th/5th) TM bands. The geometry is
radially symmetric about the cell centre, so the 32x32 0/1 map is invariant
under the full square (D4) symmetry group by construction.
"""
# EVOLVE-BLOCK-START
import math


def design(problem):
    """Return a symmetric 32x32 unit cell: a dielectric ring (annulus).

    1 = dielectric (epsilon 12), 0 = air.
    The ring has an outer radius r_out and inner radius r_in (in units of the
    lattice constant; the cell spans [-0.5, 0.5]). A thick annulus gives strong
    modulation at the Fourier orders that control the 4th/5th TM gap.
    """
    size = problem["pixels"]
    r_out = 0.46   # outer radius of the dielectric ring
    r_in = 0.20    # inner (air) radius of the ring
    r_out2 = r_out * r_out
    r_in2 = r_in * r_in

    cell = []
    for i in range(size):
        row = []
        # pixel centre coordinate, cell spans [-0.5, 0.5]
        yc = (i + 0.5) / size - 0.5
        for j in range(size):
            xc = (j + 0.5) / size - 0.5
            r2 = xc * xc + yc * yc
            # dielectric if inside the annulus
            row.append(1 if (r_in2 <= r2 <= r_out2) else 0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
