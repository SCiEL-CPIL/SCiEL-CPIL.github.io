    def axial_to_cartesian(axial_list, sep=1.01):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > 0:
            centroid = np.mean(pts, axis=0)
            pts = centroid + (pts - centroid) * sep
        return pts
    