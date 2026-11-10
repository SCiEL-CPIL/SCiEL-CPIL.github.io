import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell with a dielectric diamond.
    The diamond shape aligns with the square lattice symmetry and provides
    strong scattering at the Brillouin zone boundaries, aiming to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]
    center = (size - 1) / 2.0  # 15.5
    
    # Radius of the diamond in pixels (Manhattan distance)
    # A radius of 12-14 is typically effective for mid-band gaps.
    radius = 13.0
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Manhattan distance from center
            dist = abs(i - center) + abs(j - center)
            if dist < radius:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
